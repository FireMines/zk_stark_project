// src/root.rs

use std::error::Error;
use std::time::Instant;

use csv::ReaderBuilder;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use winterfell::{
    AcceptableOptions, Air, AirContext, Assertion, AuxRandElements, BatchingMethod, ByteWriter, CompositionPoly,
    CompositionPolyTrace, ConstraintCompositionCoefficients, DefaultConstraintCommitment, DefaultConstraintEvaluator,
    DefaultTraceLde, EvaluationFrame, FieldExtension, PartitionOptions, ProofOptions, Prover, StarkDomain, Trace,
    TraceInfo, TracePolyTable, TraceTable, TransitionConstraintDegree,
};
use winterfell::crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree};
use winterfell::math::{FieldElement, StarkField, ToElements};
use winterfell::math::fields::f128::BaseElement as Felt;
use winter_utils::Serializable;

// Global constants (matching ZoKrates)
pub const AC: usize = 6; // number of activations (layers)
pub const FE: usize = 9; // number of features per activation
pub const C: usize = 8;  // number of clients

// ----------------------- HELPER FUNCTIONS -----------------------------

/// Convert an f64 value to our field element using a scaling factor of 1e6.
pub fn f64_to_felt(x: f64) -> Felt {
    Felt::new((x * 1e6).round() as u128)
}

/// Reads a CSV file and extracts features and label.
/// If a row has 46 columns, it extracts columns 19–27 as features and column 46 as the label;
/// if a row has 10 columns, it treats the first 9 columns as features and the 10th as the label.
pub fn read_dataset(file_path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for result in rdr.records() {
        let record = result?;
        if record.is_empty() {
            continue;
        }
        if record.len() == 46 {
            let row: Vec<f64> = record.iter()
                .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
                .collect();
            features.push(row[18..27].to_vec());
            labels.push(row[45]);
        } else if record.len() == 10 {
            let row: Vec<f64> = record.iter()
                .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
                .collect();
            features.push(row[..9].to_vec());
            labels.push(row[9]);
        } else {
            return Err(format!("CSV row must have 10 or 46 columns, got {}", record.len()).into());
        }
    }
    Ok((features, labels))
}

/// Splits the dataset among a given number of clients using round‑robin.
pub fn split_dataset(
    features: Vec<Vec<f64>>,
    labels: Vec<f64>,
    num_clients: usize,
) -> Vec<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut clients = vec![(Vec::new(), Vec::new()); num_clients];
    for (i, (feat, lab)) in features.into_iter().zip(labels.into_iter()).enumerate() {
        let idx = i % num_clients;
        clients[idx].0.push(feat);
        clients[idx].1.push(lab);
    }
    clients
}

/// Generates an initial model using a normal distribution (replicates Veriblock‑FL).
pub fn generate_initial_model(input_dim: usize, output_dim: usize, precision: f64) -> (Vec<Vec<Felt>>, Vec<Felt>) {
    let normal = Normal::new(0.0, precision / 5.0).unwrap();
    let mut rng = thread_rng();
    let weights: Vec<Vec<Felt>> = (0..output_dim)
        .map(|_| {
            (0..input_dim)
                .map(|_| {
                    let sample: f64 = normal.sample(&mut rng);
                    f64_to_felt(sample)
                })
                .collect()
        })
        .collect();
    let biases: Vec<Felt> = (0..output_dim)
        .map(|_| {
            let sample: f64 = normal.sample(&mut rng);
            f64_to_felt(sample)
        })
        .collect();
    (weights, biases)
}

/// Converts a scalar label to a one‑hot vector of length AC.
pub fn label_to_one_hot(label: f64, ac: usize, precision: f64) -> Vec<Felt> {
    let mut one_hot = vec![f64_to_felt(0.0); ac];
    let idx = if label < 1.0 { 0 } else { (label as usize).saturating_sub(1) };
    if idx < ac {
        one_hot[idx] = f64_to_felt(precision);
    }
    one_hot
}

/// Flattens a weight matrix and bias vector into a single vector.
pub fn flatten_state_matrix(w: &Vec<Vec<Felt>>, b: &Vec<Felt>) -> Vec<Felt> {
    let mut flat = Vec::new();
    for row in w {
        flat.extend_from_slice(row);
    }
    flat.extend_from_slice(b);
    flat
}

/// Transposes a 2D vector.
pub fn transpose(matrix: Vec<Vec<Felt>>) -> Vec<Vec<Felt>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![f64_to_felt(0.0); rows]; cols];
    for i in 0..rows {
        assert_eq!(matrix[i].len(), cols, "All rows must have equal length");
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}

// ----------------------- SIGNED ARITHMETIC FUNCTIONS -----------------------
/// Returns (a + b, sign) using a two-part representation.
/// The sign is determined by comparing to a threshold.
pub fn signed_add(
    a: Felt,
    b: Felt,
    a_sign: Felt,
    b_sign: Felt,
    max: Felt,
    threshold: Felt,
) -> (Felt, Felt) {
    let one = Felt::ONE;
    let zero = Felt::ZERO;
    let a_cleansed = if a_sign == zero { a } else { max - a + one };
    let b_cleansed = if b_sign == zero { b } else { max - b + one };
    let c = if a_sign == one && b_sign == one {
        max + one - a_cleansed - b_cleansed
    } else {
        a + b
    };
    let c_sign = if c.as_int() > threshold.as_int() { one } else { zero };
    (c, c_sign)
}

/// Returns (a - b, sign) using the signed addition function.
pub fn signed_subtract(
    a: Felt,
    a_sign: Felt,
    b: Felt,
    b_sign: Felt,
    max: Felt,
    threshold: Felt,
) -> (Felt, Felt) {
    let b_negated_sign = if b_sign == Felt::ZERO { Felt::ONE } else { Felt::ZERO };
    signed_add(a, b, a_sign, b_negated_sign, max, threshold)
}

/// Assumes divisor b is positive. Returns (a / b, sign).
pub fn signed_divide(a: Felt, a_sign: Felt, b: Felt) -> (Felt, Felt) {
    let a_val = a.as_int();
    let b_val = b.as_int();
    let res = a_val / b_val;
    let sign = a_sign; // For simplicity, keep the original sign.
    (Felt::new(res), sign)
}

// ----------------------- LOCAL TRAINING MODULE (Client‑Side Update) -----------------------

pub mod local_training {
    use super::*;
    use winterfell::{
        Air, AirContext, Assertion, ByteWriter, CompositionPolyTrace, ConstraintCompositionCoefficients,
        DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde, EvaluationFrame,
        PartitionOptions, ProofOptions, Prover, StarkDomain, Trace, TraceInfo, TracePolyTable, TraceTable,
        TransitionConstraintDegree,
    };
    use winterfell::crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree};
    use winter_utils::Serializable;

    #[derive(Clone)]
    pub struct TrainingUpdateInputs {
        pub initial: Vec<Felt>,    // Flattened initial state: weights then biases
        pub final_state: Vec<Felt>, // Flattened updated state
        pub steps: usize,
        // New fields for actual training data and parameters:
        pub x: Vec<Felt>,          // Actual input feature vector (length FE)
        pub y: Vec<Felt>,          // Actual one‑hot label (length AC)
        pub learning_rate: Felt,   // Actual learning rate
        pub precision: Felt,       // Actual precision scaling factor
    }
    
    impl Serializable for TrainingUpdateInputs {
        fn write_into<W: ByteWriter>(&self, target: &mut W) {
            for val in &self.initial {
                target.write(*val);
            }
            for val in &self.final_state {
                target.write(*val);
            }
            target.write(f64_to_felt(self.steps as f64));
        }
    }
    
    impl ToElements<Felt> for TrainingUpdateInputs {
        fn to_elements(&self) -> Vec<Felt> {
            let mut elems = self.initial.clone();
            elems.extend(self.final_state.clone());
            elems.push(f64_to_felt(self.steps as f64));
            elems
        }
    }
    
    /// AIR for the training update circuit.
    pub struct TrainingUpdateAir {
        context: AirContext<Felt>,
        pub_inputs: TrainingUpdateInputs,
    }
    
    impl Air for TrainingUpdateAir {
        type BaseField = Felt;
        type PublicInputs = TrainingUpdateInputs;
    
        fn new(trace_info: TraceInfo, pub_inputs: TrainingUpdateInputs, options: ProofOptions) -> Self {
            let state_width = pub_inputs.initial.len();
            let degrees = vec![TransitionConstraintDegree::new(1); state_width];
            let context = AirContext::new(trace_info, degrees, state_width, options);
            Self { context, pub_inputs }
        }
    
        fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField> + From<Felt>>(
            &self,
            frame: &EvaluationFrame<E>,
            _periodic_values: &[E],
            result: &mut [E],
        ) {
            // Use the actual parameters from public inputs.
            let learning_rate: E = E::from(self.pub_inputs.learning_rate);
            let precision: E = E::from(self.pub_inputs.precision);
            let fe = self.pub_inputs.x.len(); // FE
            let ac = self.pub_inputs.y.len(); // AC
            let two: E = E::from(f64_to_felt(2.0));
    
            // For each activation (layer)
            for j in 0..ac {
                let mut dot: E = E::from(f64_to_felt(0.0));
                // Compute dot product for activation j using the actual feature vector.
                for i in 0..fe {
                    let weight_index = j * fe + i;
                    let x_i: E = E::from(self.pub_inputs.x[i]);
                    dot = dot + frame.current()[weight_index] * x_i;
                }
                // Bias is stored at index ac*fe + j.
                let bias_index = ac * fe + j;
                let pred = dot / precision + frame.current()[bias_index];
                let y_val: E = E::from(self.pub_inputs.y[j]);
                let error = pred - y_val;
                // Update weights.
                for i in 0..fe {
                    let idx = j * fe + i;
                    let x_i: E = E::from(self.pub_inputs.x[i]);
                    let update_term = learning_rate * two * error * x_i / precision;
                    let expected = frame.current()[idx] - update_term;
                    result[idx] = frame.next()[idx] - expected;
                }
                // Update bias.
                let update_bias = learning_rate * two * error;
                let expected_bias = frame.current()[bias_index] - update_bias;
                result[bias_index] = frame.next()[bias_index] - expected_bias;
            }
            // Zero out any extra registers.
            for i in (ac * fe + ac)..result.len() {
                result[i] = E::from(f64_to_felt(0.0));
            }
        }
    
        fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
            let trace_len = self.trace_length();
            self.pub_inputs.final_state
                .iter()
                .enumerate()
                .map(|(i, &val)| Assertion::single(i, trace_len - 1, val))
                .collect()
        }
    
        fn context(&self) -> &AirContext<Self::BaseField> {
            &self.context
        }
    }
    
    /// Prover for the training update circuit.
    pub struct TrainingUpdateProver {
        pub options: ProofOptions,
        pub initial_w: Vec<Vec<Felt>>, // Dimensions: AC x FE
        pub initial_b: Vec<Felt>,      // Length: AC
        pub x: Vec<Felt>,              // Input features (length FE)
        pub y: Vec<Felt>,              // One‑hot label (length AC)
        pub learning_rate: Felt,
        pub precision: Felt,
        pub trace_length: usize,
    }

    impl TrainingUpdateProver {
        pub fn new(
            options: ProofOptions,
            initial_w: Vec<Vec<Felt>>,
            initial_b: Vec<Felt>,
            x: Vec<Felt>,
            y: Vec<Felt>,
            learning_rate: Felt,
            precision: Felt,
        ) -> Self {
            let real_length = 2;
            let trace_length = (real_length as usize).next_power_of_two().max(8);
            Self {
                options,
                initial_w,
                initial_b,
                x,
                y,
                learning_rate,
                precision,
                trace_length,
            }
        }
    
        pub fn build_trace(&self) -> TraceTable<Felt> {
            // Number of registers: AC*FE (weights) + AC (biases)
            let state_width = AC * FE + AC;
            // Start with the initial state.
            let mut state = super::flatten_state_matrix(&self.initial_w, &self.initial_b);
            let mut trace_rows = vec![state.clone()];
            let two = f64_to_felt(2.0);
            let fe = self.x.len();  // should equal FE
            let ac = self.y.len();  // should equal AC

            // For each additional step, simulate an update.
            for _ in 1..self.trace_length {
                let mut new_state = state.clone();
                // Process each activation (layer)
                for j in 0..ac {
                    // Compute dot product for activation j.
                    let mut dot = f64_to_felt(0.0);
                    for i in 0..fe {
                        let idx = j * fe + i;
                        dot = dot + state[idx] * self.x[i];
                    }
                    // Bias is stored at index ac*fe + j.
                    let bias_index = ac * fe + j;
                    let pred = dot / self.precision + state[bias_index];
                    let error = pred - self.y[j];
                    // Update weights.
                    for i in 0..fe {
                        let idx = j * fe + i;
                        let update_term = self.learning_rate * two * error * self.x[i] / self.precision;
                        new_state[idx] = state[idx] - update_term;
                    }
                    // Update bias.
                    new_state[bias_index] = state[bias_index] - self.learning_rate * two * error;
                }
                trace_rows.push(new_state.clone());
                state = new_state;
            }
            let transposed = super::transpose(trace_rows);
            TraceTable::init(transposed)
        }
    }
    
    impl Prover for TrainingUpdateProver {
        type BaseField = Felt;
        type Air = TrainingUpdateAir;
        type Trace = TraceTable<Felt>;
        type HashFn = Blake3_256<Felt>;
        type VC = MerkleTree<Self::HashFn>;
        type RandomCoin = DefaultRandomCoin<Self::HashFn>;
        type TraceLde<E: FieldElement<BaseField = Self::BaseField>> =
            DefaultTraceLde<E, Self::HashFn, Self::VC>;
        type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
            DefaultConstraintEvaluator<'a, Self::Air, E>;
        type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
            DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;
    
        fn get_pub_inputs(&self, _trace: &Self::Trace) -> TrainingUpdateInputs {
            let fe = self.x.len();
            let ac = self.y.len();
            let two = f64_to_felt(2.0);
            let mut state = super::flatten_state_matrix(&self.initial_w, &self.initial_b);
    
            // Simulate update steps exactly as in build_trace.
            for _ in 1..self.trace_length {
                let mut new_state = state.clone();
                for j in 0..ac {
                    let mut dot = f64_to_felt(0.0);
                    // Compute dot product for activation j using the current state.
                    for i in 0..fe {
                        let idx = j * fe + i;
                        dot = dot + state[idx] * self.x[i];
                    }
                    // Bias is stored at index ac*fe + j.
                    let bias_index = ac * fe + j;
                    let pred = dot / self.precision + state[bias_index];
                    let error = pred - self.y[j];
                    // Update each weight.
                    for i in 0..fe {
                        let idx = j * fe + i;
                        let update_term = self.learning_rate * two * error * self.x[i] / self.precision;
                        new_state[idx] = state[idx] - update_term;
                    }
                    // Update bias.
                    new_state[bias_index] = state[bias_index] - self.learning_rate * two * error;
                }
                state = new_state;
            }
    
            TrainingUpdateInputs {
                initial: super::flatten_state_matrix(&self.initial_w, &self.initial_b),
                final_state: state,
                steps: self.trace_length - 1,
                x: self.x.clone(),
                y: self.y.clone(),
                learning_rate: self.learning_rate,
                precision: self.precision,
            }
        }
    
        fn options(&self) -> &ProofOptions {
            &self.options
        }
    
        fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            trace_info: &TraceInfo,
            main_trace: &winterfell::matrix::ColMatrix<Self::BaseField>,
            domain: &StarkDomain<Self::BaseField>,
            partition_options: PartitionOptions,
        ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
            DefaultTraceLde::new(trace_info, main_trace, domain, partition_options)
        }
    
        fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            air: &'a Self::Air,
            aux_rand_elements: Option<AuxRandElements<E>>,
            composition_coefficients: ConstraintCompositionCoefficients<E>,
        ) -> Self::ConstraintEvaluator<'a, E> {
            DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
        }
    
        fn build_constraint_commitment<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            composition_poly_trace: CompositionPolyTrace<E>,
            num_constraint_composition_columns: usize,
            domain: &StarkDomain<Self::BaseField>,
            partition_options: PartitionOptions,
        ) -> (Self::ConstraintCommitment<E>, CompositionPoly<E>) {
            DefaultConstraintCommitment::new(
                composition_poly_trace,
                num_constraint_composition_columns,
                domain,
                partition_options,
            )
        }
    }
}

// ------------------- GLOBAL UPDATE MODULE (Iterative FedAvg with Averaging) -----------------------

pub mod global_update {
    use super::*;
    use winterfell::matrix::ColMatrix;
    
    #[derive(Clone)]
    pub struct GlobalUpdateInputs {
        pub global_w: Vec<Vec<Felt>>,    // initial global weights [AC x FE]
        pub global_b: Vec<Felt>,         // initial global biases [AC]
        pub new_global_w: Vec<Vec<Felt>>, // aggregated new weights [AC x FE]
        pub new_global_b: Vec<Felt>,      // aggregated new biases [AC]
        pub k: Felt,                     // scaling factor
        pub digest: Felt,                // MiMC hash digest of new model
    }
    
    impl Serializable for GlobalUpdateInputs {
        fn write_into<W: ByteWriter>(&self, target: &mut W) {
            for i in 0..AC {
                for j in 0..FE {
                    target.write(self.global_w[i][j]);
                }
            }
            for i in 0..AC {
                target.write(self.global_b[i]);
            }
            for i in 0..AC {
                for j in 0..FE {
                    target.write(self.new_global_w[i][j]);
                }
            }
            for i in 0..AC {
                target.write(self.new_global_b[i]);
            }
            target.write(self.k);
            target.write(self.digest);
        }
    }
    
    impl ToElements<Felt> for GlobalUpdateInputs {
        fn to_elements(&self) -> Vec<Felt> {
            let mut elems = Vec::new();
            for i in 0..AC {
                for j in 0..FE {
                    elems.push(self.global_w[i][j]);
                }
            }
            for i in 0..AC {
                elems.push(self.global_b[i]);
            }
            for i in 0..AC {
                for j in 0..FE {
                    elems.push(self.new_global_w[i][j]);
                }
            }
            for i in 0..AC {
                elems.push(self.new_global_b[i]);
            }
            elems.push(self.k);
            elems.push(self.digest);
            elems
        }
    }
    
    /// AIR for the global update circuit.
    pub struct GlobalUpdateAir {
        context: AirContext<Felt>,
        pub_inputs: GlobalUpdateInputs,
    }
    
    impl GlobalUpdateAir {
        /// Returns the public initial state (flattened) from the public inputs.
        fn get_public_initial_state(&self) -> Vec<Felt> {
            let mut state = Vec::new();
            for i in 0..AC {
                for j in 0..FE {
                    state.push(self.pub_inputs.global_w[i][j]);
                }
            }
            for i in 0..AC {
                state.push(self.pub_inputs.global_b[i]);
            }
            state
        }
    
        /// Returns the public final (aggregated) state (flattened) from the public inputs.
        fn get_public_new_state(&self) -> Vec<Felt> {
            let mut state = Vec::new();
            for i in 0..AC {
                for j in 0..FE {
                    state.push(self.pub_inputs.new_global_w[i][j]);
                }
            }
            for i in 0..AC {
                state.push(self.pub_inputs.new_global_b[i]);
            }
            state
        }
    }
    
    impl Air for GlobalUpdateAir {
        type BaseField = Felt;
        type PublicInputs = GlobalUpdateInputs;
    
        fn new(trace_info: TraceInfo, pub_inputs: GlobalUpdateInputs, options: ProofOptions) -> Self {
            // total width = state registers + 1 (time)
            let state_width = AC * FE + AC;
            let degrees = vec![TransitionConstraintDegree::new(1); state_width];
            let context = AirContext::new(trace_info, degrees, state_width, options);
            Self { context, pub_inputs }
        }
    
        fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField> + From<Felt>>(
            &self,
            frame: &EvaluationFrame<E>,
            _periodic_values: &[E],
            result: &mut [E],
        ) {
            let n_minus_one = E::from(Felt::new((self.trace_length() - 1) as u128)).inv();
            let public_initial = self.get_public_initial_state();
            let public_final = self.get_public_new_state();
            let expected_deltas: Vec<E> = public_initial
                .iter()
                .zip(public_final.iter())
                .map(|(init, fin)| E::from(*fin - *init) * n_minus_one)
                .collect();
            for i in 0..result.len() {
                let actual_delta = frame.next()[i] - frame.current()[i];
                result[i] = actual_delta - expected_deltas[i];
            }
        }
    
        fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
            let public_final = self.get_public_new_state();
            (0..public_final.len())
                .map(|i| Assertion::single(i, self.trace_length() - 1, public_final[i]))
                .collect()
        }
    
        fn context(&self) -> &AirContext<Self::BaseField> {
            &self.context
        }
    }
        
    /// GlobalUpdateProver using iterative FedAvg aggregation with averaging.
    pub struct GlobalUpdateProver {
        pub options: ProofOptions,
        pub global_w: Vec<Vec<Felt>>,
        pub global_w_sign: Vec<Vec<Felt>>,
        pub global_b: Vec<Felt>,
        pub global_b_sign: Vec<Felt>,
        pub local_w: Vec<Vec<Vec<Felt>>>,    // [C][AC][FE]
        pub local_w_sign: Vec<Vec<Vec<Felt>>>, // [C][AC][FE]
        pub local_b: Vec<Vec<Felt>>,           // [C][AC]
        pub local_b_sign: Vec<Vec<Felt>>,      // [C][AC]
        pub k: Felt,
        pub trace_length: usize, // padded to a power‑of‑two (min 8)
    }
    
    impl GlobalUpdateProver {
        pub fn new(
            options: ProofOptions,
            global_w: Vec<Vec<Felt>>,
            global_w_sign: Vec<Vec<Felt>>,
            global_b: Vec<Felt>,
            global_b_sign: Vec<Felt>,
            local_w: Vec<Vec<Vec<Felt>>>,
            local_w_sign: Vec<Vec<Vec<Felt>>>,
            local_b: Vec<Vec<Felt>>,
            local_b_sign: Vec<Vec<Felt>>,
            k: Felt,
        ) -> Self {
            let padded_rows: usize = std::cmp::max(local_w.len() + 1, 8).next_power_of_two();
            Self {
                options,
                global_w,
                global_w_sign,
                global_b,
                global_b_sign,
                local_w,
                local_w_sign,
                local_b,
                local_b_sign,
                k,
                trace_length: padded_rows,
            }
        }
    
        fn flatten_state(w: &Vec<Vec<Felt>>, b: &Vec<Felt>) -> Vec<Felt> {
            let mut flat = Vec::new();
            for row in w {
                flat.extend_from_slice(row);
            }
            flat.extend_from_slice(b);
            flat
        }
    
        /// Computes the complete trace of iterative state updates.
        /// Returns a vector of state rows, one per update (including the initial state).
        pub fn compute_iterative_trace(&self) -> Vec<Vec<Felt>> {
            let max = Felt::new(u128::MAX);
            let threshold = Felt::new(1u128 << 120);
            let mut current_w = self.global_w.clone();
            let mut current_w_sign = self.global_w_sign.clone();
            let mut current_b = self.global_b.clone();
            let mut current_b_sign = self.global_b_sign.clone();
            let mut trace_rows = Vec::with_capacity(self.local_w.len() + 1);
            trace_rows.push(Self::flatten_state(&current_w, &current_b));
    
            // Process each client update iteratively.
            for client in 0..self.local_w.len() {
                // Update weights.
                for i in 0..AC {
                    for j in 0..FE {
                        let (temp, temp_sign) = signed_subtract(
                            self.local_w[client][i][j],
                            self.local_w_sign[client][i][j],
                            current_w[i][j],
                            current_w_sign[i][j],
                            max,
                            threshold,
                        );
                        let (temp2, temp2_sign) = signed_divide(temp, temp_sign, self.k);
                        let (new_val, new_sign) = signed_add(
                            current_w[i][j],
                            current_w_sign[i][j],
                            temp2,
                            temp2_sign,
                            max,
                            threshold,
                        );
                        current_w[i][j] = new_val;
                        current_w_sign[i][j] = new_sign;
                    }
                }
                // Update biases.
                for i in 0..AC {
                    let (temp, temp_sign) = signed_subtract(
                        self.local_b[client][i],
                        self.local_b_sign[client][i],
                        current_b[i],
                        current_b_sign[i],
                        max,
                        threshold,
                    );
                    let (temp2, temp2_sign) = signed_divide(temp, temp_sign, self.k);
                    let (new_bias, new_bias_sign) = signed_add(
                        current_b[i],
                        current_b_sign[i],
                        temp2,
                        temp2_sign,
                        max,
                        threshold,
                    );
                    current_b[i] = new_bias;
                    current_b_sign[i] = new_bias_sign;
                }
                // Record the updated state.
                trace_rows.push(Self::flatten_state(&current_w, &current_b));
            }
            trace_rows
        }
    
        /// Builds the STARK trace by padding the complete iterative trace to a power-of‑two length.
        pub fn build_trace(&self) -> TraceTable<Felt> {
            let mut trace_rows = self.compute_iterative_trace();
            let num_rows = trace_rows.len();
            let padded_rows = num_rows.next_power_of_two().max(8);
            while trace_rows.len() < padded_rows {
                trace_rows.push(trace_rows.last().unwrap().clone());
            }
            let transposed = super::transpose(trace_rows);
            TraceTable::init(transposed)
        }
    }
    
    impl Prover for GlobalUpdateProver {
        type BaseField = Felt;
        type Air = GlobalUpdateAir;
        type Trace = TraceTable<Felt>;
        type HashFn = Blake3_256<Felt>;
        type VC = MerkleTree<Self::HashFn>;
        type RandomCoin = DefaultRandomCoin<Self::HashFn>;
        type TraceLde<E: FieldElement<BaseField = Self::BaseField>> =
            DefaultTraceLde<E, Self::HashFn, Self::VC>;
        type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
            DefaultConstraintEvaluator<'a, Self::Air, E>;
        type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
            DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;
    
        fn get_pub_inputs(&self, _trace: &Self::Trace) -> GlobalUpdateInputs {
            let trace_rows = self.compute_iterative_trace();
            let final_state = trace_rows.last().unwrap().clone();
            let new_global_w: Vec<Vec<Felt>> = final_state[..(AC * FE)]
                .chunks(FE)
                .map(|chunk| chunk.to_vec())
                .collect();
            let new_global_b: Vec<Felt> = final_state[(AC * FE)..].to_vec();
            GlobalUpdateInputs {
                global_w: self.global_w.clone(),
                global_b: self.global_b.clone(),
                new_global_w,
                new_global_b,
                k: self.k,
                digest: f64_to_felt(0.0),
            }
        }
    
        fn options(&self) -> &ProofOptions {
            &self.options
        }
    
        fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            trace_info: &TraceInfo,
            main_trace: &winterfell::matrix::ColMatrix<Self::BaseField>,
            domain: &StarkDomain<Self::BaseField>,
            partition_options: PartitionOptions,
        ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
            DefaultTraceLde::new(trace_info, main_trace, domain, partition_options)
        }
    
        fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            air: &'a Self::Air,
            aux_rand_elements: Option<AuxRandElements<E>>,
            composition_coefficients: ConstraintCompositionCoefficients<E>,
        ) -> Self::ConstraintEvaluator<'a, E> {
            DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
        }
    
        fn build_constraint_commitment<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            composition_poly_trace: CompositionPolyTrace<E>,
            num_constraint_composition_columns: usize,
            domain: &StarkDomain<Self::BaseField>,
            partition_options: PartitionOptions,
        ) -> (Self::ConstraintCommitment<E>, CompositionPoly<E>) {
            DefaultConstraintCommitment::new(
                composition_poly_trace,
                num_constraint_composition_columns,
                domain,
                partition_options,
            )
        }
    }
    
    pub fn mimc_cipher(input: Felt, round_constant: Felt, z: Felt) -> Felt {
        let mut inp = input;
        for _ in 0..64 {
            let a = inp + round_constant + z;
            inp = <Felt as FieldElement>::exp(a, 7);
        }
        inp + z
    }
    
    pub fn mimc_hash_matrix(w: &[Vec<Felt>], b: &[Felt], round_constants: &[Felt]) -> Felt {
        let mut z = f64_to_felt(0.0);
        for i in 0..w.len() {
            for j in 0..w[i].len() {
                let rc = round_constants[j % round_constants.len()];
                z = mimc_cipher(w[i][j], rc, z);
            }
            let rc = round_constants[i % round_constants.len()];
            z = mimc_cipher(b[i], rc, z);
        }
        z
    }
}

// ------------------------- MAIN FUNCTION -------------------------

use local_training::TrainingUpdateProver;
use global_update::{GlobalUpdateProver, mimc_hash_matrix};

fn main() -> Result<(), Box<dyn Error>> {
    // --- DATASET LOADING ---
    let (features, labels) = read_dataset("devices/edge_device/data/train.txt")?;
    let client_data = split_dataset(features, labels, C); // 8 clients
    
    // --- CLIENT SIDE: Compute local training updates.
    println!("--- Client Training Updates ---");
    let client_proof_options = ProofOptions::new(
        40, 16, 21, FieldExtension::None, 16, 7,
        BatchingMethod::Algebraic, BatchingMethod::Algebraic,
    );
    let mut total_client_time = 0;
    let mut client_final_reps = Vec::new();
    for (i, (client_features, client_labels)) in client_data.iter().enumerate() {
        if client_features.is_empty() {
            return Err("Client data is empty".into());
        }
        // Use the first sample from each client. Expect each sample to have FE features.
        let sample = &client_features[0];
        if sample.len() != FE {
            return Err(format!("CSV row must have {} feature columns, got {}", FE, sample.len()).into());
        }
        let x: Vec<Felt> = sample.iter().map(|&v| f64_to_felt(v)).collect();
        // Convert scalar label into one‑hot vector.
        let label_val = client_labels[0];
        let y: Vec<Felt> = label_to_one_hot(label_val, AC, 1e6);
        // Generate initial model using real data (replicating Veriblock‑FL).
        let (init_w, init_b) = generate_initial_model(FE, AC, 10000.0);
        let learning_rate = f64_to_felt(1000.0);
        let precision = f64_to_felt(1e6);
        let start = Instant::now();
        let training_prover = TrainingUpdateProver::new(
            client_proof_options.clone(),
            init_w.clone(),
            init_b.clone(),
            x,
            y,
            learning_rate,
            precision,
        );
        let trace = training_prover.build_trace();
        let _ = training_prover.prove(trace.clone())?;
        let elapsed = start.elapsed().as_millis();
        println!("Client {}: Proof generation time: {} ms", i + 1, elapsed);
        total_client_time += elapsed;
        let pub_inputs = training_prover.get_pub_inputs(&trace);
        // Use the first element of the final state as the representative update.
        client_final_reps.push(pub_inputs.final_state[0]);
    }
    println!(
        "Average client update time: {} ms",
        total_client_time / (client_data.len() as u128)
    );
    
    // --- GLOBAL UPDATE: FedAvg Aggregation ---
    println!("\n--- Global Update Example ---");
    let (global_w, global_b) = generate_initial_model(FE, AC, 10000.0);
    let global_w_sign: Vec<Vec<Felt>> = vec![vec![f64_to_felt(0.0); FE]; AC];
    let global_b_sign: Vec<Felt> = vec![f64_to_felt(0.0); AC];
    
    let mut local_w = Vec::new();
    let mut local_w_sign = Vec::new();
    let mut local_b = Vec::new();
    let mut local_b_sign = Vec::new();
    for rep in client_final_reps.iter() {
        let client_val = (*rep).as_int() as f64 / 1e6;
        let client_w_mat: Vec<Vec<Felt>> = vec![vec![f64_to_felt(client_val); FE]; AC];
        let client_w_sign_mat: Vec<Vec<Felt>> = client_w_mat.iter().map(|row| row.iter().map(|_| f64_to_felt(0.0)).collect()).collect();
        let client_b_vec: Vec<Felt> = vec![f64_to_felt(client_val); AC];
        let client_b_sign_vec: Vec<Felt> = client_b_vec.iter().map(|_| f64_to_felt(0.0)).collect();
        local_w.push(client_w_mat);
        local_w_sign.push(client_w_sign_mat);
        local_b.push(client_b_vec);
        local_b_sign.push(client_b_sign_vec);
    }
    let k = f64_to_felt(C as f64);
    let aggregator_prover = GlobalUpdateProver::new(
        client_proof_options.clone(),
        global_w,
        global_w_sign,
        global_b,
        global_b_sign,
        local_w,
        local_w_sign,
        local_b,
        local_b_sign,
        k,
    );
    let trace = aggregator_prover.build_trace();
    println!("Aggregator trace built with {} rows.", trace.length());
    let trace_rows = aggregator_prover.compute_iterative_trace();
    let final_state = trace_rows.last().unwrap().clone();
    let new_global_w: Vec<Vec<Felt>> = final_state[..(AC * FE)]
        .chunks(FE)
        .map(|chunk| chunk.to_vec())
        .collect();
    let new_global_b: Vec<Felt> = final_state[(AC * FE)..].to_vec();
    let round_constants: Vec<Felt> = (0..64).map(|_| f64_to_felt(42.0)).collect();
    let flat_new_w: Vec<Felt> = new_global_w.iter().flatten().cloned().collect();
    let computed_digest = mimc_hash_matrix(&new_global_w, &new_global_b, &round_constants);
    let mut pub_inputs = aggregator_prover.get_pub_inputs(&trace);
    pub_inputs.digest = computed_digest;
    println!("Global (old) weights: {:?}", pub_inputs.global_w);
    println!("Global (old) biases:  {:?}", pub_inputs.global_b);
    println!("New Global (aggregated) weights: {:?}", pub_inputs.new_global_w);
    println!("New Global (aggregated) biases:  {:?}", pub_inputs.new_global_b);
    println!("Computed MiMC digest: {:?}", computed_digest);
    let start = Instant::now();
    let proof = aggregator_prover.prove(trace)?;
    println!("Global update proof generated in {} ms", start.elapsed().as_millis());
    //println!("Global update proof (hex): {}", hex::encode(proof.to_bytes()));
    let acceptable_options = AcceptableOptions::OptionSet(vec![client_proof_options]);
    match winterfell::verify::<global_update::GlobalUpdateAir, Blake3_256<Felt>, DefaultRandomCoin<Blake3_256<Felt>>, MerkleTree<Blake3_256<Felt>>>(
        proof,
        pub_inputs,
        &acceptable_options,
    ) {
        Ok(_) => println!("Global update proof verified successfully."),
        Err(e) => {
            println!("Global update proof verification failed: {}", e);
            match e {
                winterfell::VerifierError::RandomCoinError => println!("randomcoin"),
                winterfell::VerifierError::InconsistentBaseField => println!("InconsistentBaseField"),
                winterfell::VerifierError::UnsupportedFieldExtension(_) => println!("UnsupportedFieldExtension"),
                winterfell::VerifierError::ProofDeserializationError(_) => println!("ProofDeserializationError"),
                winterfell::VerifierError::InconsistentOodConstraintEvaluations => println!("InconsistentOodConstraintEvaluations"),
                winterfell::VerifierError::TraceQueryDoesNotMatchCommitment => println!("TraceQueryDoesNotMatchCommitment"),
                winterfell::VerifierError::ConstraintQueryDoesNotMatchCommitment => println!("ConstraintQueryDoesNotMatchCommitment"),
                winterfell::VerifierError::QuerySeedProofOfWorkVerificationFailed => println!("QuerySeedProofOfWorkVerificationFailed"),
                winterfell::VerifierError::FriVerificationFailed(verifier_error) => println!("FriVerificationFailed"),
                winterfell::VerifierError::InsufficientConjecturedSecurity(_, _) => println!("InsufficientConjecturedSecurity"),
                winterfell::VerifierError::InsufficientProvenSecurity(_, _) => println!("InsufficientProvenSecurity"),
                winterfell::VerifierError::UnacceptableProofOptions => println!("UnacceptableProofOptions"),
            }
        } ,
    }
    Ok(())
}
