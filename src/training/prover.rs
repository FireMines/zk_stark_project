// src/training/prover.rs

use rand::Rng;
use winterfell::{
    AuxRandElements, CompositionPoly, CompositionPolyTrace, ConstraintCompositionCoefficients, DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde, PartitionOptions, ProofOptions, Prover, StarkDomain, Trace, TraceInfo, TracePolyTable, TraceTable
};
use winterfell::crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree};
use winterfell::math::FieldElement;
use winterfell::math::fields::f128::BaseElement as Felt;

use crate::training::air::{TrainingUpdateAir, TrainingUpdateInputs};
use crate::helper::{
    backward_propagation_layer, forward_propagation_layer, mse_prime,
    split_state_with_sign, transpose,
};

/// Prover for the masked zk‑STARK of one SGD step with batch processing.
pub struct TrainingUpdateProver {
    options: ProofOptions,
    initial_w: Vec<Vec<Felt>>, // AC × FE
    initial_b: Vec<Felt>,      // AC
    w_sign: Vec<Vec<Felt>>,    // AC × FE
    b_sign: Vec<Felt>,         // AC
    x_batch: Vec<Vec<Felt>>,   // BS × FE (batch of feature vectors)
    x_batch_sign: Vec<Vec<Felt>>, // BS × FE (sign vectors for features)
    y_batch: Vec<Vec<Felt>>,   // BS × AC (batch of one-hot labels)
    learning_rate: Felt,
    precision: Felt,
    trace_length: usize,
    batch_size: usize,
}

impl TrainingUpdateProver {
    pub fn new(
        options: ProofOptions,
        initial_w: Vec<Vec<Felt>>,
        initial_b: Vec<Felt>,
        w_sign: Vec<Vec<Felt>>,
        b_sign: Vec<Felt>,
        x_batch: Vec<Vec<Felt>>,        // BS × FE
        x_batch_sign: Vec<Vec<Felt>>,   // BS × FE  
        y_batch: Vec<Vec<Felt>>,        // BS × AC
        learning_rate: Felt,
        precision: Felt,
        batch_size: usize,
    ) -> Self {
        let ac = initial_b.len();
        let fe = initial_w[0].len();
        
        // Validate inputs
        assert_eq!(x_batch.len(), batch_size, "x_batch size doesn't match batch_size");
        assert_eq!(x_batch_sign.len(), batch_size, "x_batch_sign size doesn't match batch_size");
        assert_eq!(y_batch.len(), batch_size, "y_batch size doesn't match batch_size");
        
        let state_cells = ac * fe + ac;
        // Trace length needs to accommodate batch processing
        let trace_length = (2 * state_cells * batch_size).next_power_of_two().max(16);

        Self {
            options,
            initial_w,
            initial_b,
            w_sign,
            b_sign,
            x_batch,
            x_batch_sign,
            y_batch,
            learning_rate,
            precision,
            trace_length,
            batch_size,
        }
    }

    /// Build the masked trace for batch processing
    pub fn build_trace(&self) -> TraceTable<Felt> {
        let ac = self.initial_b.len();
        let fe = self.initial_w[0].len();
        let state_cells = ac * fe + ac;
        let flat_len = state_cells * 2;

        // flatten raw initial [v0,s0,v1,s1,…]
        let mut raw = Vec::with_capacity(flat_len);
        for (row, srow) in self.initial_w.iter().zip(&self.w_sign) {
            for (&v, &s) in row.iter().zip(srow) {
                raw.push(v);
                raw.push(s);
            }
        }
        for (&v, &s) in self.initial_b.iter().zip(&self.b_sign) {
            raw.push(v);
            raw.push(s);
        }

        let mut rng = rand::thread_rng();
        // sample initial mask and masked state
        let mut mask = (0..flat_len)
            .map(|_| Felt::new(rng.gen::<u64>() as u128))
            .collect::<Vec<_>>();
        let mut masked = raw
            .iter()
            .zip(&mask)
            .map(|(r, m)| *r + *m)
            .collect::<Vec<_>>();

        let mut rows = Vec::with_capacity(self.trace_length);
        // push row 0 = [masked || mask]
        rows.push(masked.iter().chain(mask.iter()).cloned().collect());

        // Process batch samples sequentially within the trace
        for step in 1..self.trace_length {
            // recover (w,b) and their signs from raw
            let (mut w, mut b, mut w_s, mut b_s) =
                split_state_with_sign(&raw, ac, fe);

            // Process one sample from the batch if within batch size
            if step <= self.batch_size {
                let sample_idx = step - 1;
                
                // forward pass for this sample
                let (out, out_s) = forward_propagation_layer(
                    &w, &b, &self.x_batch[sample_idx], 
                    &w_s, &b_s, &self.x_batch_sign[sample_idx], 
                    self.precision,
                );
                
                // compute error for this sample
                let (mut err, mut err_s) = mse_prime(
                    &self.y_batch[sample_idx], &out, &out_s, self.precision
                );
                
                // backward pass for this sample
                let (w2, b2, w2_s, b2_s) = backward_propagation_layer(
                    &mut w, &mut b, &self.x_batch[sample_idx], &mut err,
                    self.learning_rate, self.precision,
                    &mut w_s, &mut b_s,
                    &self.x_batch_sign[sample_idx], &mut err_s,
                );

                // flatten next raw
                raw.clear();
                for (rvec, svec) in w2.iter().zip(&w2_s) {
                    for (&v, &sv) in rvec.iter().zip(svec) {
                        raw.push(v);
                        raw.push(sv);
                    }
                }
                for (&v, &sv) in b2.iter().zip(&b2_s) {
                    raw.push(v);
                    raw.push(sv);
                }
            }
            // If step > batch_size, just maintain the same state

            // sample new mask
            mask = (0..flat_len)
                .map(|_| Felt::new(rng.gen::<u64>() as u128))
                .collect();
            // compute masked next
            masked = raw
                .iter()
                .zip(&mask)
                .map(|(r, m)| *r + *m)
                .collect();

            // push [masked || mask]
            rows.push(masked.iter().chain(mask.iter()).cloned().collect());
        }

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
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;
    type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;

    fn get_pub_inputs(&self, trace: &Self::Trace) -> TrainingUpdateInputs {
        let rows = trace.length();
        let cols = trace.width();
        let half = cols / 2;

        // extract masked initial (row 0) and masked final (row N)
        let initial_masked: Vec<Felt> = (0..half).map(|c| trace.get(c, 0)).collect();
        let final_masked:   Vec<Felt> = (0..half).map(|c| trace.get(c, rows - 1)).collect();

        TrainingUpdateInputs {
            initial_masked,
            final_masked,
            steps:         self.trace_length - 1,
            x_batch:       self.x_batch.clone(),
            y_batch:       self.y_batch.clone(),
            learning_rate: self.learning_rate,
            precision:     self.precision,
            batch_size:    self.batch_size,
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