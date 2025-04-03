use std::time::Instant;

use csv::ReaderBuilder;
use winterfell::{
    Air, AirContext, Assertion, ByteWriter, EvaluationFrame, FieldExtension, ProofOptions,
    Prover, Trace, TraceInfo, TraceTable, TransitionConstraintDegree, AcceptableOptions,
    crypto::{DefaultRandomCoin, MerkleTree, hashers::Blake3_256},
    DefaultTraceLde, DefaultConstraintEvaluator, DefaultConstraintCommitment,
    matrix::ColMatrix, BatchingMethod, StarkDomain, TracePolyTable, PartitionOptions,
    CompositionPolyTrace, CompositionPoly, ConstraintCompositionCoefficients, AuxRandElements,
};
use winterfell::math::{FieldElement, ToElements};
use winterfell::math::fields::f128::BaseElement as Felt;
use winter_utils::Serializable;

/// Helper: Convert an f64 value to our field element Felt
/// (Assumes a fixed scaling factor, here 1e6, adjust as needed)
fn f64_to_felt(x: f64) -> Felt {
    // Multiply by scaling factor and round to an integer.
    // (In practice, choose the scaling factor to suit your dataset.)
    Felt::new((x * 1e6).round() as u128)
}

/// Read a CSV file where each row is a list of f64 values.
fn read_dataset(file_path: &str) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(file_path)?;
    let mut dataset = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let row: Vec<f64> = record
            .iter()
            .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
            .collect();
        dataset.push(row);
    }
    Ok(dataset)
}

/// Divide the dataset among a given number of clients.
/// For simplicity we assign rows round-robin.
fn split_dataset(dataset: Vec<Vec<f64>>, num_clients: usize) -> Vec<Vec<Vec<f64>>> {
    let mut clients: Vec<Vec<Vec<f64>>> = vec![vec![]; num_clients];
    for (i, row) in dataset.into_iter().enumerate() {
        clients[i % num_clients].push(row);
    }
    clients
}

/// For this simple example, assume that each client uses a single feature:
/// we take the last column as the "label" and ignore the other columns.
/// We then define:
///   initial_weight is fixed (say 100), and
///   gradient = initial_weight - (average(label) over the client's rows)
/// so that final_weight = initial_weight - gradient = average(label).
fn compute_update_for_client(data: &[Vec<f64>], initial: f64) -> (Felt, Vec<Felt>) {
    let sum: f64 = data.iter().map(|row| row.last().unwrap_or(&0.0)).sum();
    let avg = sum / (data.len() as f64);
    let gradient = initial - avg;
    // Convert to field elements using the helper.
    (f64_to_felt(initial), vec![f64_to_felt(gradient)])
}

/// --- Training Update Circuit (Client Side) --- ///
mod training {
    use super::*;
    #[derive(Clone)]
    pub struct TrainingInputs {
        pub initial: Felt,
        pub final_weight: Felt,
        pub steps: usize,
    }
    
    impl Serializable for TrainingInputs {
        fn write_into<W: ByteWriter>(&self, target: &mut W) {
            target.write(self.initial);
            target.write(self.final_weight);
            target.write(Felt::new(self.steps as u128));
        }
    }
    
    impl ToElements<Felt> for TrainingInputs {
        fn to_elements(&self) -> Vec<Felt> {
            vec![self.initial, self.final_weight, Felt::new(self.steps as u128)]
        }
    }
    
    /// AIR for training update.
    /// Enforces that the trace obeys the update rule: weight_next = weight - gradient.
    pub struct TrainingAir {
        context: AirContext<Felt>,
        initial: Felt,
        final_weight: Felt,
    }
    
    impl Air for TrainingAir {
        type BaseField = Felt;
        type PublicInputs = TrainingInputs;
    
        fn new(trace_info: TraceInfo, pub_inputs: TrainingInputs, options: ProofOptions) -> Self {
            let degrees = vec![TransitionConstraintDegree::new(1)];
            let context = AirContext::new(trace_info, degrees, 2, options);
            Self {
                context,
                initial: pub_inputs.initial,
                final_weight: pub_inputs.final_weight,
            }
        }
    
        // We assume that E is exactly our base field type (Felt).
        fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            frame: &EvaluationFrame<E>,
            _periodic_values: &[E],
            result: &mut [E],
        ) {
            // Enforce: weight - gradient - weight_next == 0.
            result[0] = frame.current()[0] - frame.current()[1] - frame.next()[0];
        }
    
        fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
            let trace_len = self.trace_length();
            vec![
                Assertion::single(0, 0, self.initial),
                Assertion::single(0, trace_len - 1, self.final_weight),
            ]
        }
    
        fn context(&self) -> &AirContext<Self::BaseField> {
            &self.context
        }
    }
    
    /// Prover for the training update circuit.
    pub struct TrainingProver {
        pub options: ProofOptions,
        pub gradients: Vec<Felt>,
        pub trace_length: usize,
        pub initial: Felt,
    }
    
    impl TrainingProver {
        pub fn new(options: ProofOptions, initial: Felt, gradients: Vec<Felt>) -> Self {
            let real_length: usize = gradients.len() + 1;
            let mut trace_length = real_length.next_power_of_two();
            if trace_length < 8 {
                trace_length = 8;
            }
            Self { options, gradients, trace_length, initial }
        }
    
        pub fn build_trace(&self) -> TraceTable<Felt> {
            let mut weight_trace = Vec::with_capacity(self.trace_length);
            let mut gradient_trace = Vec::with_capacity(self.trace_length);
            weight_trace.push(self.initial);
            for &g in &self.gradients {
                let prev = *weight_trace.last().unwrap();
                let next = prev - g;
                weight_trace.push(next);
                gradient_trace.push(g);
            }
            // Pad the trace so that its length is a power of two.
            while weight_trace.len() < self.trace_length {
                let prev = *weight_trace.last().unwrap();
                weight_trace.push(prev);
                gradient_trace.push(Felt::new(0));
            }
            if gradient_trace.len() < weight_trace.len() {
                gradient_trace.push(Felt::new(0));
            }
            TraceTable::init(vec![weight_trace, gradient_trace])
        }
    }
    
    impl Prover for TrainingProver {
        type BaseField = Felt;
        type Air = TrainingAir;
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
    
        fn get_pub_inputs(&self, trace: &Self::Trace) -> TrainingInputs {
            TrainingInputs {
                initial: trace.get(0, 0),
                final_weight: trace.get(0, trace.length() - 1),
                steps: self.gradients.len(),
            }
        }
    
        fn options(&self) -> &ProofOptions {
            &self.options
        }
    
        fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            trace_info: &TraceInfo,
            main_trace: &ColMatrix<Self::BaseField>,
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

/// --- Global Update (Aggregation) Circuit --- ///
mod global_update {
    use super::*;
    
    #[derive(Clone)]
    pub struct GlobalUpdateInputs {
        pub global: Felt,     // current global parameter
        pub new_global: Felt, // updated global parameter
        pub k: Felt,          // scaling factor
        pub digest: Felt,     // expected MiMC hash digest
    }
    
    impl Serializable for GlobalUpdateInputs {
        fn write_into<W: ByteWriter>(&self, target: &mut W) {
            target.write(self.global);
            target.write(self.new_global);
            target.write(self.k);
            target.write(self.digest);
        }
    }
    
    impl ToElements<Felt> for GlobalUpdateInputs {
        fn to_elements(&self) -> Vec<Felt> {
            vec![self.global, self.new_global, self.k, self.digest]
        }
    }
    
    pub struct GlobalUpdateAir {
        context: AirContext<Felt>,
        pub_inputs: GlobalUpdateInputs,
    }
    
    impl Air for GlobalUpdateAir {
        type BaseField = Felt;
        type PublicInputs = GlobalUpdateInputs;
    
        fn new(trace_info: TraceInfo, pub_inputs: GlobalUpdateInputs, options: ProofOptions) -> Self {
            let degrees = vec![TransitionConstraintDegree::new(1)];
            let context = AirContext::new(trace_info, degrees, 2, options);
            Self { context, pub_inputs }
        }
    
        fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            frame: &EvaluationFrame<E>,
            _periodic_values: &[E],
            result: &mut [E],
        ) {
            // Retrieve scaling factor k from public inputs and use it directly.
            let k = E::from(self.pub_inputs.k);
            let g = frame.current()[0];
            let u = frame.current()[1];
            let g_next = frame.next()[0];
            // Constraint: k * new_global - k * global - update = 0.
            result[0] = k * g_next - k * g - u;
        }
    
        fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
            let trace_len = self.trace_length();
            vec![
                Assertion::single(0, 0, self.pub_inputs.global),
                Assertion::single(0, trace_len - 1, self.pub_inputs.new_global),
            ]
        }
    
        fn context(&self) -> &AirContext<Self::BaseField> {
            &self.context
        }
    }
    
    pub struct GlobalUpdateProver {
        pub options: ProofOptions,
        pub global: Felt,
        pub update: Felt, // new_global - global
        pub trace_length: usize,
        pub k: Felt,
    }
    
    impl GlobalUpdateProver {
        pub fn new(options: ProofOptions, global: Felt, update: Felt, k: Felt) -> Self {
            let real_length: usize = 2;
            let mut trace_length = real_length.next_power_of_two();
            if trace_length < 8 {
                trace_length = 8;
            }
            Self { options, global, update, trace_length, k }
        }
    
        pub fn build_trace(&self) -> TraceTable<Felt> {
            let new_global = self.global + self.update;
            // Build a 2-row trace (one real transition) and then pad.
            let mut value_trace = vec![self.global, new_global];
            let mut update_trace = vec![self.update, Felt::new(0)];
            while value_trace.len() < self.trace_length {
                let last_val = *value_trace.last().unwrap();
                value_trace.push(last_val);
                update_trace.push(Felt::new(0));
            }
            TraceTable::init(vec![value_trace, update_trace])
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
    
        fn get_pub_inputs(&self, trace: &Self::Trace) -> GlobalUpdateInputs {
            let trace_len = trace.length();
            GlobalUpdateInputs {
                global: trace.get(0, 0),
                new_global: trace.get(0, trace_len - 1),
                k: self.k,
                digest: Felt::new(0), // This will be set externally.
            }
        }
    
        fn options(&self) -> &ProofOptions {
            &self.options
        }
    
        fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            trace_info: &TraceInfo,
            main_trace: &ColMatrix<Self::BaseField>,
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
    
    // --- MiMC Cipher and Hash Functions for Aggregation --- //
    pub fn mimc_cipher(input: Felt, round_constant: Felt, z: Felt) -> Felt {
        // Use the FieldElement trait’s exponentiation method.
        // Here we use exp with exponent 7.
        <Felt as FieldElement>::exp(input + round_constant + z, 7) + z
    }
    
    pub fn mimc_hash_matrix(w: &[Vec<Felt>], b: &[Felt], round_constants: &[Felt]) -> Felt {
        let mut z = Felt::new(0);
        let ac = w.len();
        let fe = if ac > 0 { w[0].len() } else { 0 };
        for i in 0..ac {
            for j in 0..fe {
                let rc = round_constants[j % round_constants.len()];
                z = mimc_cipher(w[i][j], rc, z);
            }
            let rc = round_constants[i % round_constants.len()];
            z = mimc_cipher(b[i], rc, z);
        }
        z
    }
}

/// --- MAIN: Proof Generation, Dataset Loading & Verification --- ///
use training::TrainingProver;
use global_update::{GlobalUpdateProver, mimc_hash_matrix as global_mimc_hash};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- DATASET LOADING ---
    // Assume your merged dataset is stored in "devices/edge_device/data/train.txt"
    let dataset = read_dataset("devices/edge_device/data/train.txt")?;
    // Split dataset among clients (e.g. 8 clients for this example)
    let client_data = split_dataset(dataset, 8);
    
    // --- CLIENT SIDE: Compute local updates based on the dataset ---
    // For each client, we assume:
    // - The model starts with an initial weight (e.g. 100.0)
    // - The client computes the average of the last column (label) and uses that to derive a gradient.
    let initial_value = 100.0;
    let mut client_final_weights = Vec::new();
    
    // Proof options for the circuits.
    let client_proof_options = ProofOptions::new(
        40, 16, 21, FieldExtension::None, 16, 7,
        BatchingMethod::Algebraic, BatchingMethod::Algebraic,
    );
    
    println!("--- Client Training Updates ---");
    for (i, data) in client_data.iter().enumerate() {
        // Compute initial and gradient for this client.
        let (init, gradients) = compute_update_for_client(data, initial_value);
        let training_prover = TrainingProver::new(client_proof_options.clone(), init, gradients);
        let trace = training_prover.build_trace();
        let pub_inputs = training_prover.get_pub_inputs(&trace);
        let start = Instant::now();
        // Generate training proof (proof content not used further in this example)
        let _proof = training_prover.prove(trace).expect("Training proof generation failed");
        let duration = start.elapsed().as_millis();
        println!(
            "Client {}: Initial = {:?}, Final = {:?} (Proof generated in {} ms)",
            i + 1, pub_inputs.initial, pub_inputs.final_weight, duration
        );
        client_final_weights.push(pub_inputs.final_weight);
    }
    
    // --- GLOBAL UPDATE: Aggregation ---
    println!("\n--- Global Update Example ---");
    // Assume the current global model is 100 (in field representation)
    let global = f64_to_felt(initial_value);
    // Sum the final weights from all clients.
    let sum: Felt = client_final_weights.iter().fold(Felt::new(0), |acc, &w| acc + w);
    // The update is the difference between the sum and the current global.
    let update = sum - global;
    // Use a scaling factor k (here we set k = 1 for simplicity; adjust as needed)
    let k = Felt::new(1);
    let global_update_prover = GlobalUpdateProver::new(client_proof_options.clone(), global, update, k);
    let trace = global_update_prover.build_trace();
    let mut pub_inputs = global_update_prover.get_pub_inputs(&trace);
    // Prepare MiMC round constants (for simplicity, using constant 42 for all rounds)
    let round_constants: Vec<Felt> = (0..64).map(|_| Felt::new(42)).collect();
    let computed_digest = global_mimc_hash(&vec![vec![pub_inputs.new_global]], &vec![Felt::new(0)], &round_constants);
    pub_inputs.digest = computed_digest;
    println!("Global (old): {:?}", pub_inputs.global);
    println!("New Global:   {:?}", pub_inputs.new_global);
    println!("Computed MiMC digest: {:?}", computed_digest);
    
    let start = Instant::now();
    let proof = global_update_prover.prove(trace).expect("Global update proof generation failed");
    println!("Global update proof generated in {} ms", start.elapsed().as_millis());
    //println!("Global update proof (hex): {}", hex::encode(proof.to_bytes()));
    
    let acceptable_options = AcceptableOptions::OptionSet(vec![client_proof_options.clone()]);
    match winterfell::verify::<global_update::GlobalUpdateAir, Blake3_256<Felt>, DefaultRandomCoin<Blake3_256<Felt>>, MerkleTree<Blake3_256<Felt>>>(
        proof,
        pub_inputs,
        &acceptable_options,
    ) {
        Ok(_) => println!("Global update proof verified successfully."),
        Err(e) => println!("Global update proof verification failed: {}", e),
    }
    
    Ok(())
}
