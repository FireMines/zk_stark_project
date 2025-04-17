// src/training/prover.rs

use crate::helper::{
    backward_propagation_layer, flatten_state_matrix, flatten_with_sign, forward_propagation_layer, mse_prime, split_state_with_sign, transpose, unflatten_state_matrix, unflatten_values_only
};
use crate::training::air::{TrainingUpdateAir, TrainingUpdateInputs};
use winterfell::{
    AuxRandElements, CompositionPoly, CompositionPolyTrace, ConstraintCompositionCoefficients, DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde, PartitionOptions, ProofOptions, Prover, StarkDomain, Trace, TraceInfo, TracePolyTable, TraceTable
};
use winterfell::crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree};
use winterfell::math::FieldElement;
use winterfell::math::fields::f128::BaseElement as Felt;

/// Prover for the training update circuit.
pub struct TrainingUpdateProver {
    pub options: ProofOptions,
    pub initial_w: Vec<Vec<Felt>>, // Dimensions: AC x FE
    pub initial_b: Vec<Felt>,      // Length: AC
    pub w_sign: Vec<Vec<Felt>>,    // Dimensions: AC x FE
    pub b_sign: Vec<Felt>,         // Length: AC
    pub x: Vec<Felt>,              // Input features (length FE)
    pub x_sign: Vec<Felt>,         // Input signs (length FE)
    pub y: Vec<Felt>,              // One‑hot label (length AC)
    pub learning_rate: Felt,
    pub precision: Felt,
    pub trace_length: usize,
}

impl TrainingUpdateProver {
    /// Updated constructor now accepts sign vectors for weights, biases, and inputs.
    pub fn new(
        options: ProofOptions,
        initial_w: Vec<Vec<Felt>>,
        initial_b: Vec<Felt>,
        w_sign: Vec<Vec<Felt>>,
        b_sign: Vec<Felt>,
        x: Vec<Felt>,
        x_sign: Vec<Felt>,
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
            w_sign,
            b_sign,
            x,
            x_sign,
            y,
            learning_rate,
            precision,
            trace_length,
        }
    }
    pub fn debug_activation(
        &self,
        w: &Vec<Vec<Felt>>,
        b: &Vec<Felt>,
        x: &Vec<Felt>,
        error: &Vec<Felt>,
        learning_rate: Felt,
        pr: Felt,
        activation: usize,
    ) {
        println!("--- Debug Activation {} ---", activation);
        let fe = x.len();
        let mut dot = Felt::ZERO;
        for i in 0..fe {
            dot = dot + w[activation][i] * x[i];
        }
        let pred = dot / pr + b[activation];
        println!("  dot = {:?}", dot);
        println!("  pred = {:?}", pred);
        println!("  error = {:?}", error[activation]);
    }
    

    /// Builds the execution trace that covers both forward and backward propagation.
    pub fn build_trace(&self) -> TraceTable<Felt> {
        // Determine dimensions.
        let fe = self.x.len();
        let ac = self.y.len();

        // ---- Dimension sanity checks -----------------------------------------
        assert_eq!(
            self.initial_w.len(),
            ac,
            "initial_w has {} rows, but AC (=y.len()) is {}", 
            self.initial_w.len(), ac
        );
        assert_eq!(
            self.initial_b.len(),
            ac,
            "initial_b length ({}) ≠ AC ({})",
            self.initial_b.len(), ac
        );
        assert_eq!(
            self.w_sign.len(),
            ac,
            "w_sign has {} rows, but AC ({}) expected",
            self.w_sign.len(), ac
        );
        for (row_idx, row) in self.w_sign.iter().enumerate() {
            assert_eq!(
                row.len(),
                fe,
                "w_sign[{}] has length {}, but FE (=x.len()) is {}",
                row_idx,
                row.len(),
                fe
            );
        }
        // ----------------------------------------------------------------------

    
        // Start with the flattened initial state.
        let mut state = flatten_with_sign(&self.initial_w,&self.w_sign,
            &self.initial_b,&self.b_sign);
        let mut trace_rows = vec![state.clone()];
    
        // Create mutable copies of the sign vectors stored in the prover.
        let mut current_w_sign = self.w_sign.clone();
        let mut current_b_sign = self.b_sign.clone();
        // Also, create mutable copies for x and x_sign.
        let mut x = self.x.clone();
        let mut x_sign = self.x_sign.clone();
    
        // For each step in the trace, simulate one training update.
        for _ in 1..self.trace_length {
            // Unflatten the current state into weights and biases.
            let (mut current_w, mut current_b,
                mut current_w_sign, mut current_b_sign) =
                   split_state_with_sign(&state, ac, fe);    
            // === Forward Propagation ===
            // Use the updated sign vectors (current_w_sign and current_b_sign) here.
            let (output, out_sign) = forward_propagation_layer(
                &current_w,
                &current_b,
                &self.x,
                &current_w_sign, // using updated weight sign vector
                &current_b_sign, // using updated bias sign vector
                &self.x_sign,
                self.precision,
            );
            // 'output' holds the predictions computed from current_w, current_b, and self.x.
    
            // === Loss Derivative (MSE Prime) ===
            // Compute the derivative (error) between the true output (self.y) and the prediction.
            let (error, error_sign) = mse_prime(&self.y, &output, &out_sign, self.precision);
    
            #[cfg(debug_assertions)]
            {
                // Debug print for each activation.
                //for j in 0..ac {
                    //self.debug_activation(&current_w, &current_b, &self.x, &error, self.learning_rate, self.precision, j);
                //}
            }
    
            // For backward propagation, create mutable copies of error and error_sign.
            let mut error_mut = error.clone();
            let mut error_sign_mut = error_sign.clone();
    
            // === Backward Propagation ===
            // Pass mutable references to current_w, current_b, x, and the sign vectors.
            let (updated_w, updated_b, updated_w_sign, updated_b_sign) =
                backward_propagation_layer(
                    &mut current_w,
                    &mut current_b,
                    &self.x, // here using self.x as it isn’t updated
                    &mut error_mut,
                    self.learning_rate,
                    self.precision,
                    &mut current_w_sign,
                    &mut current_b_sign,
                    &self.x_sign,
                    &mut error_sign_mut,
                );
    
            // Re-flatten the updated parameters to create the new state.
            state = flatten_with_sign(&updated_w,&updated_w_sign,
                &updated_b,&updated_b_sign);
            trace_rows.push(state.clone());
    
            // **Critical change:** Update the sign vectors for the next iteration.
            current_w_sign = updated_w_sign;
            current_b_sign = updated_b_sign;
        }
    
        // Transpose the trace rows (if required by your AIR) and initialize the TraceTable.
        let transposed = transpose(trace_rows);
    
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

    /// Instead of re-simulating the update loop, we extract the final state directly
    /// from the trace. This ensures that our public inputs exactly match the trace used
    /// in verification.
    fn get_pub_inputs(&self, trace: &Self::Trace) -> TrainingUpdateInputs {
        // Assuming the trace is a matrix with:
        // - state_width columns (length of the flattened state)
        // - trace_length rows (one for each time step)
        let trace_len = trace.length();
        let state_width = trace.width(); // adjust if your API differs

        // Extract the final state (last row of the trace).
        let mut final_state = Vec::with_capacity(state_width);
        for col in 0..state_width {
            final_state.push(trace.get(col, trace_len - 1));
        }

        TrainingUpdateInputs {
            initial: flatten_with_sign(
                &self.initial_w, &self.w_sign,
                &self.initial_b, &self.b_sign
            ),
            final_state,
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


