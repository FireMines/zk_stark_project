// aggregation_seperated/aggregation/prover.rs

use crate::aggregation::air::{GlobalUpdateAir, GlobalUpdateInputs};
use crate::helper::{AC, FE, transpose, get_round_constants, mimc_hash_matrix};
use winterfell::{
    AuxRandElements, CompositionPoly, CompositionPolyTrace, ConstraintCompositionCoefficients,
    DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde, PartitionOptions, ProofOptions,
    Prover, TraceInfo, TraceTable,
};
use winterfell::crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree};
use winterfell::math::FieldElement;
use winterfell::math::fields::f128::BaseElement as Felt;
use rand::Rng;

pub struct GlobalUpdateProver {
    options: ProofOptions,

    // real (unmasked) initial model
    raw_global_w: Vec<Vec<Felt>>,
    raw_global_b: Vec<Felt>,

    // local updates
    local_w: Vec<Vec<Vec<Felt>>>,
    local_b: Vec<Vec<Felt>>,

    k: Felt,
    trace_length: usize,

    // random blinding
    blinding: Vec<Felt>,

    // masked initial model
    masked_global_w: Vec<Vec<Felt>>,
    masked_global_b: Vec<Felt>,
}

impl GlobalUpdateProver {
    // Flatten / unflatten helpers
    fn flatten_state(w: &Vec<Vec<Felt>>, b: &Vec<Felt>) -> Vec<Felt> {
        let mut flat = Vec::new();
        for row in w {
            flat.extend_from_slice(row);
        }
        flat.extend_from_slice(b);
        flat
    }
    fn unflatten_state(state: &[Felt], ac: usize, fe: usize) -> (Vec<Vec<Felt>>, Vec<Felt>) {
        let weights: Vec<Vec<Felt>> = (0..ac)
            .map(|i| state[i*fe..(i*fe+fe)].to_vec())
            .collect();
        let biases = state[(ac*fe)..].to_vec();
        (weights, biases)
    }

    pub fn new(
        options: ProofOptions,
        raw_global_w: Vec<Vec<Felt>>,
        raw_global_b: Vec<Felt>,
        local_w: Vec<Vec<Vec<Felt>>>,
        local_b: Vec<Vec<Felt>>,
        k: Felt,
    ) -> Self {
        let uns_padded_steps = local_w.len() + 2;
        let padded_rows = uns_padded_steps.next_power_of_two().max(8);
        let d = AC * FE + AC;

        // sample a random blinding
        let mut rng = rand::thread_rng();
        let blinding: Vec<Felt> = (0..d).map(|_| {
            let r: u64 = rng.gen();
            Felt::new(r as u128)
        }).collect();

        // compute masked initial
        let raw_flat = Self::flatten_state(&raw_global_w, &raw_global_b);
        let masked_flat: Vec<Felt> = raw_flat.iter()
            .zip(blinding.iter())
            .map(|(raw_val, mask)| *raw_val + *mask)
            .collect();
        let (masked_global_w, masked_global_b) = Self::unflatten_state(&masked_flat, AC, FE);

        Self {
            options,
            raw_global_w,
            raw_global_b,
            local_w,
            local_b,
            k,
            trace_length: padded_rows,
            blinding,
            masked_global_w,
            masked_global_b,
        }
    }

    /// The masked iterative trace. row0 is the masked old model, row(i+1) = row(i) + delta
    /// where delta = (L_i - raw_global) / c.
    pub fn compute_iterative_trace_augmented(&self) -> Vec<Vec<Felt>> {
        let num_transitions = self.local_w.len();
        let uns_padded_steps = num_transitions + 2;
        let d_state = AC * FE + AC;

        let mut trace_rows = Vec::with_capacity(uns_padded_steps);

        // row0: [masked old model || 0]
        let mut row0 = Self::flatten_state(&self.masked_global_w, &self.masked_global_b);
        row0.extend(vec![Felt::ZERO; d_state]);
        trace_rows.push(row0.clone());

        let mut current_masked = Self::flatten_state(&self.masked_global_w, &self.masked_global_b);

        let raw_flat = Self::flatten_state(&self.raw_global_w, &self.raw_global_b);

        for i in 0..num_transitions {
            // Flatten local update
            let local_w_flat: Vec<Felt> = self.local_w[i].iter().flat_map(|r| r.clone()).collect();
            let mut l = local_w_flat;
            l.extend(self.local_b[i].clone());

            // compute delta = (L - raw_global) / c
            let delta: Vec<Felt> = raw_flat.iter()
                .zip(l.iter())
                .map(|(g0, l)| (*l - *g0) / self.k)
                .collect();

            // next_masked = current_masked + delta
            let next_masked: Vec<Felt> = current_masked.iter()
                .zip(delta.iter())
                .map(|(cm, d)| *cm + *d)
                .collect();

            // the update portion is (L - raw_global)
            let update_stored: Vec<Felt> = raw_flat.iter()
                .zip(l.iter())
                .map(|(g0, l)| *l - *g0)
                .collect();

            let mut row = next_masked.clone();
            row.extend(update_stored);
            trace_rows.push(row.clone());
            current_masked = next_masked;
        }

        // final row: [S_final_masked || 0]
        let extra_row = [current_masked.clone(), vec![Felt::ZERO; d_state]].concat();
        trace_rows.push(extra_row.clone());

        // pad if needed
        while trace_rows.len() < self.trace_length {
            let last = trace_rows.last().unwrap().clone();
            trace_rows.push(last);
        }
        trace_rows
    }

    /// Build the trace table from the masked iterative rows
    pub fn build_trace(&self) -> TraceTable<Felt> {
        println!("Building aggregation trace for {} clients...", self.local_w.len());
        let trace_rows = self.compute_iterative_trace_augmented();
        println!("Aggregation trace built: {} rows", trace_rows.len());
        TraceTable::init(transpose(trace_rows))
    }
    
    /// Extract the masked final state -> masked old/new states in the public inputs
    pub fn get_pub_inputs(&self, _trace: &<GlobalUpdateProver as Prover>::Trace) -> GlobalUpdateInputs {
        let num_transitions = self.local_w.len();
        let uns_padded_steps = num_transitions + 2;

        // the final row is the last unpadded row
        let trace_rows = self.compute_iterative_trace_augmented();
        let final_row = trace_rows[uns_padded_steps - 1].clone();
        let d_state = AC * FE + AC;
        let final_masked_state = &final_row[..d_state];

        // unflatten the final masked
        let (new_w_masked, new_b_masked) = Self::unflatten_state(final_masked_state, AC, FE);

        // compute digest
        let round_constants = get_round_constants();
        let digest = mimc_hash_matrix(&new_w_masked, &new_b_masked, &round_constants);

        // public old = masked_global_w, masked_global_b
        GlobalUpdateInputs {
            global_w: self.masked_global_w.clone(),
            global_b: self.masked_global_b.clone(),
            new_global_w: new_w_masked,
            new_global_b: new_b_masked,
            k: self.k,
            digest,
            steps: uns_padded_steps,
        }
    }
}

// implement Prover trait for the aggregator as usual
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
        self.get_pub_inputs(trace)
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }

    fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &winterfell::matrix::ColMatrix<Self::BaseField>,
        domain: &winterfell::StarkDomain<Self::BaseField>,
        partition_options: PartitionOptions,
    ) -> (Self::TraceLde<E>, winterfell::TracePolyTable<E>) {
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
        domain: &winterfell::StarkDomain<Self::BaseField>,
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
