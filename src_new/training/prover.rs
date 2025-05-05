use winterfell::{
    AuxRandElements, CompositionPoly, CompositionPolyTrace, ConstraintCompositionCoefficients,
    DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde, PartitionOptions,
    ProofOptions, Prover, StarkDomain, Trace, TraceInfo, TracePolyTable, TraceTable,
};
use winterfell::crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree};
use winterfell::math::{FieldElement, StarkField};
use winterfell::math::fields::f128::BaseElement as Felt;

use crate::training::air::{TrainingUpdateAir, TrainingUpdateInputs};
use crate::helper::{
    AC, FE, 
    forward_propagation_batch, backward_propagation_batch, mse_prime, 
    split_state_with_sign, transpose
};

pub struct TrainingUpdateProver {
    options: ProofOptions,
    initial_w: Vec<Vec<Felt>>,
    initial_b: Vec<Felt>,
    w_sign: Vec<Vec<Felt>>,
    b_sign: Vec<Felt>,
    x: Vec<Felt>,
    x_sign: Vec<Felt>,
    y: Vec<Felt>,
    y_sign: Vec<Felt>,
    batch_size: usize,
    learning_rate: Felt,
    precision: Felt,
    trace_length: usize,
}

impl TrainingUpdateProver {
    pub fn new(
        options: ProofOptions,
        initial_w: Vec<Vec<Felt>>,
        initial_b: Vec<Felt>,
        w_sign: Vec<Vec<Felt>>,
        b_sign: Vec<Felt>,
        x: Vec<Felt>,
        x_sign: Vec<Felt>,
        y: Vec<Felt>,
        y_sign: Vec<Felt>,
        learning_rate: Felt,
        precision: Felt,
    ) -> Self {
        // Validation checks
        assert_eq!(initial_w.len(), AC, "Initial weights rows must match AC");
        assert_eq!(initial_w[0].len(), FE, "Initial weights columns must match FE");
        assert_eq!(initial_b.len(), AC, "Initial bias length must match AC");
        assert_eq!(w_sign.len(), AC, "Weight sign rows must match AC");
        assert_eq!(w_sign[0].len(), FE, "Weight sign columns must match FE");
        assert_eq!(b_sign.len(), AC, "Bias sign length must match AC");

        // Calculate batch size and validate related dimensions
        let batch_size = x.len() / FE;
        assert_eq!(x.len(), batch_size * FE, "x length must be a multiple of FE");
        assert_eq!(x_sign.len(), x.len(), "x_sign length must match x length");
        assert_eq!(y.len(), batch_size * AC, "y length must match batch_size * AC");
        assert_eq!(y_sign.len(), y.len(), "y_sign length must match y length");

        // Calculate trace length consistently
        let state_cells = AC * FE + AC;
        let trace_length = (2 * state_cells).next_power_of_two().max(8);

        println!("Initializing prover with batch_size={}, trace_length={}", batch_size, trace_length);

        Self {
            options,
            initial_w,
            initial_b,
            w_sign,
            b_sign,
            x,
            x_sign,
            y,
            y_sign,
            batch_size,
            learning_rate,
            precision,
            trace_length,
        }
    }

    pub fn build_trace(&self) -> TraceTable<Felt> {
        println!("Building trace with trace_length={}", self.trace_length);
        
        // Fall back to single-sample implementation for batch_size=1
        if self.batch_size == 1 {
            return self.build_trace_single_sample();
        }
        
        // Calculate the number of parameters
        let num_weights = AC * FE;
        let num_biases = AC;
        let num_weight_signs = AC * FE;
        let num_bias_signs = AC;
        let num_params = num_weights + num_biases + num_weight_signs + num_bias_signs;
        
        // For masked state, we need twice as many columns
        let state_width = num_params * 2;
        let mut rows = Vec::with_capacity(self.trace_length);
        
        // Initialize first row with masked values
        let mut current_state = Vec::with_capacity(state_width);
        
        // Generate random masks for initial state
        let mut masks = Vec::with_capacity(num_params);
        for _ in 0..num_params {
            // In a real implementation, use a secure random number generator
            masks.push(Felt::new(123456)); 
        }
        
        // Create masked initial state: value+mask, mask for each parameter
        let mut param_idx = 0;
        
        // Add masked weights
        for i in 0..AC {
            for j in 0..FE {
                // Weight value
                current_state.push(self.initial_w[i][j] + masks[param_idx]);
                current_state.push(masks[param_idx]);
                param_idx += 1;
            }
        }
        
        // Add masked biases
        for i in 0..AC {
            current_state.push(self.initial_b[i] + masks[param_idx]);
            current_state.push(masks[param_idx]);
            param_idx += 1;
        }
        
        // Add masked weight signs
        for i in 0..AC {
            for j in 0..FE {
                if param_idx < masks.len() {
                    current_state.push(self.w_sign[i][j] + masks[param_idx]);
                    current_state.push(masks[param_idx]);
                    param_idx += 1;
                }
            }
        }
        
        // Add masked bias signs
        for i in 0..AC {
            if param_idx < masks.len() {
                current_state.push(self.b_sign[i] + masks[param_idx]);
                current_state.push(masks[param_idx]);
                param_idx += 1;
            }
        }
        
        rows.push(current_state.clone());
        
        // Track progress
        let progress_interval = (self.trace_length / 10).max(1);
        
        // Compute subsequent states
        for step in 1..self.trace_length {
            if step % progress_interval == 0 || step == 1 {
                println!("Building trace: step {}/{} ({:.1}%)", 
                         step, self.trace_length, (step as f64 / self.trace_length as f64) * 100.0);
            }
            
            // Extract current weights and biases (unmask them)
            let mut w = vec![vec![Felt::ZERO; FE]; AC];
            let mut b = vec![Felt::ZERO; AC];
            let mut w_s = vec![vec![Felt::ZERO; FE]; AC];
            let mut b_s = vec![Felt::ZERO; AC];
            
            // Unmask weights and biases
            param_idx = 0;
            
            // Extract weights
            for i in 0..AC {
                for j in 0..FE {
                    if param_idx*2+1 < current_state.len() {
                        w[i][j] = current_state[param_idx*2] - current_state[param_idx*2+1];
                        param_idx += 1;
                    }
                }
            }
            
            // Extract biases
            for i in 0..AC {
                if param_idx*2+1 < current_state.len() {
                    b[i] = current_state[param_idx*2] - current_state[param_idx*2+1];
                    param_idx += 1;
                }
            }
            
            // Extract weight signs
            for i in 0..AC {
                for j in 0..FE {
                    if param_idx*2+1 < current_state.len() {
                        w_s[i][j] = current_state[param_idx*2] - current_state[param_idx*2+1];
                        param_idx += 1;
                    }
                }
            }
            
            // Extract bias signs
            for i in 0..AC {
                if param_idx*2+1 < current_state.len() {
                    b_s[i] = current_state[param_idx*2] - current_state[param_idx*2+1];
                    param_idx += 1;
                }
            }
            
            // Instead of processing each sample separately, use batch functions
            
            // 1. Forward propagation on the entire batch
            let (out, out_s) = forward_propagation_batch(
                &w, &b, &w_s, &b_s, &self.x_sign, &self.x,
                self.batch_size, self.precision
            );
            
            // 2. Calculate error for the batch
            let (err, err_s) = mse_prime(
                &self.y, &out, &out_s, self.precision
            );
            
            // 3. Backward propagation on the entire batch
            let (w_updated, b_updated, w_s_updated, b_s_updated) = backward_propagation_batch(
                &w, &b, &w_s, &b_s, &self.x_sign, &self.x,
                &err_s, &err, self.batch_size, 
                self.learning_rate, self.precision
            );
            
            // Generate new masks for this step
            let mut new_masks = Vec::with_capacity(num_params);
            for _ in 0..num_params {
                // In a real implementation, use a secure random number generator
                new_masks.push(Felt::new(987654 + step as u128)); 
            }
            
            // Build new masked state
            let mut new_state = Vec::with_capacity(state_width);
            
            // Mask updated weights
            param_idx = 0;
            for i in 0..AC {
                for j in 0..FE {
                    if param_idx < new_masks.len() {
                        new_state.push(w_updated[i][j] + new_masks[param_idx]);
                        new_state.push(new_masks[param_idx]);
                        param_idx += 1;
                    }
                }
            }
            
            // Mask updated biases
            for i in 0..AC {
                if param_idx < new_masks.len() {
                    new_state.push(b_updated[i] + new_masks[param_idx]);
                    new_state.push(new_masks[param_idx]);
                    param_idx += 1;
                }
            }
            
            // Mask updated weight signs
            for i in 0..AC {
                for j in 0..FE {
                    if param_idx < new_masks.len() {
                        new_state.push(w_s_updated[i][j] + new_masks[param_idx]);
                        new_state.push(new_masks[param_idx]);
                        param_idx += 1;
                    }
                }
            }
            
            // Mask updated bias signs
            for i in 0..AC {
                if param_idx < new_masks.len() {
                    new_state.push(b_s_updated[i] + new_masks[param_idx]);
                    new_state.push(new_masks[param_idx]);
                    param_idx += 1;
                }
            }
            
            rows.push(new_state.clone());
            current_state = new_state;
        }
    
        println!("Trace built with {} rows", rows.len());
        TraceTable::init(transpose(rows))
    }
    
    fn build_trace_single_sample(&self) -> TraceTable<Felt> {
        println!("Using single-sample trace builder for batch_size=1");
        
        // For single sample, just use the regular approach but still use our batch functions
        // by treating it as a batch of size 1
        
        // Calculate the number of parameters
        let num_weights = AC * FE;
        let num_biases = AC;
        let num_weight_signs = AC * FE;
        let num_bias_signs = AC;
        let num_params = num_weights + num_biases + num_weight_signs + num_bias_signs;
        
        // For masked state, we need twice as many columns
        let state_width = num_params * 2;
        let mut rows = Vec::with_capacity(self.trace_length);
        
        // Generate random masks for initial state
        let mut masks = Vec::with_capacity(num_params);
        for _ in 0..num_params {
            // In a real implementation, use a secure random number generator
            masks.push(Felt::new(123456)); 
        }
        
        // Initialize first row with masked values
        let mut current_state = Vec::with_capacity(state_width);
        
        // Create masked initial state: value+mask, mask for each parameter
        let mut param_idx = 0;
        
        // Add masked weights
        for i in 0..AC {
            for j in 0..FE {
                // Weight value
                current_state.push(self.initial_w[i][j] + masks[param_idx]);
                current_state.push(masks[param_idx]);
                param_idx += 1;
            }
        }
        
        // Add masked biases
        for i in 0..AC {
            current_state.push(self.initial_b[i] + masks[param_idx]);
            current_state.push(masks[param_idx]);
            param_idx += 1;
        }
        
        // Add masked weight signs
        for i in 0..AC {
            for j in 0..FE {
                if param_idx < masks.len() {
                    current_state.push(self.w_sign[i][j] + masks[param_idx]);
                    current_state.push(masks[param_idx]);
                    param_idx += 1;
                }
            }
        }
        
        // Add masked bias signs
        for i in 0..AC {
            if param_idx < masks.len() {
                current_state.push(self.b_sign[i] + masks[param_idx]);
                current_state.push(masks[param_idx]);
                param_idx += 1;
            }
        }
        
        rows.push(current_state.clone());
        
        // Compute steps
        for step in 1..self.trace_length {
            if step % 10 == 0 {
                println!("Single-sample trace: step {}/{}", step, self.trace_length);
            }
            
            // Extract current weights and biases (unmask them)
            let mut w = vec![vec![Felt::ZERO; FE]; AC];
            let mut b = vec![Felt::ZERO; AC];
            let mut w_s = vec![vec![Felt::ZERO; FE]; AC];
            let mut b_s = vec![Felt::ZERO; AC];
            
            // Unmask weights and biases
            param_idx = 0;
            
            // Extract weights
            for i in 0..AC {
                for j in 0..FE {
                    if param_idx*2+1 < current_state.len() {
                        w[i][j] = current_state[param_idx*2] - current_state[param_idx*2+1];
                        param_idx += 1;
                    }
                }
            }
            
            // Extract biases
            for i in 0..AC {
                if param_idx*2+1 < current_state.len() {
                    b[i] = current_state[param_idx*2] - current_state[param_idx*2+1];
                    param_idx += 1;
                }
            }
            
            // Extract weight signs
            for i in 0..AC {
                for j in 0..FE {
                    if param_idx*2+1 < current_state.len() {
                        w_s[i][j] = current_state[param_idx*2] - current_state[param_idx*2+1];
                        param_idx += 1;
                    }
                }
            }
            
            // Extract bias signs
            for i in 0..AC {
                if param_idx*2+1 < current_state.len() {
                    b_s[i] = current_state[param_idx*2] - current_state[param_idx*2+1];
                    param_idx += 1;
                }
            }
            
            // Forward propagation on the single sample as a batch of size 1
            let (out, out_s) = forward_propagation_batch(
                &w, &b, &w_s, &b_s, &self.x_sign, &self.x,
                1, self.precision
            );
            
            // Calculate error for the single sample
            let (err, err_s) = mse_prime(
                &self.y, &out, &out_s, self.precision
            );
            
            // Backward propagation on the single sample as a batch of size 1
            let (w_updated, b_updated, w_s_updated, b_s_updated) = backward_propagation_batch(
                &w, &b, &w_s, &b_s, &self.x_sign, &self.x,
                &err_s, &err, 1, 
                self.learning_rate, self.precision
            );
            
            // Generate new masks for this step
            let mut new_masks = Vec::with_capacity(num_params);
            for _ in 0..num_params {
                // In a real implementation, use a secure random number generator
                new_masks.push(Felt::new(987654 + step as u128)); 
            }
            
            // Build new masked state
            let mut new_state = Vec::with_capacity(state_width);
            
            // Mask updated weights
            param_idx = 0;
            for i in 0..AC {
                for j in 0..FE {
                    if param_idx < new_masks.len() {
                        new_state.push(w_updated[i][j] + new_masks[param_idx]);
                        new_state.push(new_masks[param_idx]);
                        param_idx += 1;
                    }
                }
            }
            
            // Mask updated biases
            for i in 0..AC {
                if param_idx < new_masks.len() {
                    new_state.push(b_updated[i] + new_masks[param_idx]);
                    new_state.push(new_masks[param_idx]);
                    param_idx += 1;
                }
            }
            
            // Mask updated weight signs
            for i in 0..AC {
                for j in 0..FE {
                    if param_idx < new_masks.len() {
                        new_state.push(w_s_updated[i][j] + new_masks[param_idx]);
                        new_state.push(new_masks[param_idx]);
                        param_idx += 1;
                    }
                }
            }
            
            // Mask updated bias signs
            for i in 0..AC {
                if param_idx < new_masks.len() {
                    new_state.push(b_s_updated[i] + new_masks[param_idx]);
                    new_state.push(new_masks[param_idx]);
                    param_idx += 1;
                }
            }
            
            rows.push(new_state.clone());
            current_state = new_state;
        }
        
        println!("Single-sample trace built with {} rows", rows.len());
        TraceTable::init(transpose(rows))
    }
}

impl Prover for TrainingUpdateProver {
    type BaseField = Felt;
    type Air = TrainingUpdateAir;
    type Trace = TraceTable<Felt>;
    type HashFn = Blake3_256<Felt>;
    type VC = MerkleTree<Self::HashFn>;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;

    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> = DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> = DefaultConstraintEvaluator<'a, Self::Air, E>;
    type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> = DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;

    fn get_pub_inputs(&self, trace: &Self::Trace) -> TrainingUpdateInputs {
        let steps = trace.length() - 1;
        let width = trace.width();
        
        // Extract masked initial and final states
        let mut initial_masked = Vec::with_capacity(width);
        let mut final_masked = Vec::with_capacity(width);
        
        for c in 0..width {
            initial_masked.push(trace.get(c, 0));
            final_masked.push(trace.get(c, steps));
        }

        println!("Created public inputs with masked states: steps={}, width={}", steps, width);

        TrainingUpdateInputs {
            initial_state: initial_masked,
            final_state: final_masked,
            steps,
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
        info: &TraceInfo,
        main: &winterfell::matrix::ColMatrix<Self::BaseField>,
        domain: &StarkDomain<Self::BaseField>,
        po: PartitionOptions,
    ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
        DefaultTraceLde::new(info, main, domain, po)
    }

    fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        air: &'a Self::Air,
        aux: Option<AuxRandElements<E>>,
        comp: ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E> {
        DefaultConstraintEvaluator::new(air, aux, comp)
    }

    fn build_constraint_commitment<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        trace: CompositionPolyTrace<E>,
        num_cols: usize,
        domain: &StarkDomain<Self::BaseField>,
        po: PartitionOptions,
    ) -> (Self::ConstraintCommitment<E>, CompositionPoly<E>) {
        DefaultConstraintCommitment::new(trace, num_cols, domain, po)
    }
}