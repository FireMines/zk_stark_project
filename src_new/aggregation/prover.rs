// src/aggregation/prover.rs

use crate::aggregation::air::{GlobalUpdateAir, GlobalUpdateInputs};
use crate::helper::{AC, FE, f64_to_felt, transpose};
use winterfell::{
    ProofOptions, Prover, TraceInfo, TraceTable, PartitionOptions,
    DefaultTraceLde, DefaultConstraintEvaluator, DefaultConstraintCommitment, CompositionPolyTrace, CompositionPoly,
    ConstraintCompositionCoefficients, AuxRandElements,
};
use winterfell::crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree};
use winterfell::math::FieldElement;
use winterfell::math::fields::f128::BaseElement as Felt;

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
    pub trace_length: usize,
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
        let padded_rows = std::cmp::max(local_w.len() + 1, 8).next_power_of_two();
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
        for row in w { flat.extend_from_slice(row); }
        flat.extend_from_slice(b);
        flat
    }

    pub fn compute_iterative_trace(&self) -> Vec<Vec<Felt>> {
        let num_clients = Felt::new(self.local_w.len() as u128);
        let mut current_w = self.global_w.clone();
        let mut current_b = self.global_b.clone();
        let mut trace_rows = Vec::with_capacity(self.local_w.len() + 1);
        trace_rows.push(Self::flatten_state(&current_w, &current_b));
        
        for client in 0..self.local_w.len() {
            for i in 0..AC {
                for j in 0..FE {
                    let diff = self.local_w[client][i][j] - current_w[i][j];
                    let update = diff / num_clients;
                    current_w[i][j] = current_w[i][j] + update;
                }
            }
            for i in 0..AC {
                let diff = self.local_b[client][i] - current_b[i];
                let update = diff / num_clients;
                current_b[i] = current_b[i] + update;
            }
            trace_rows.push(Self::flatten_state(&current_w, &current_b));
        }
        trace_rows
    }

    pub fn build_trace(&self) -> TraceTable<Felt> {
        let mut trace_rows = self.compute_iterative_trace();
        let num_rows = trace_rows.len();
        let padded_rows = num_rows.next_power_of_two().max(8);
        while trace_rows.len() < padded_rows {
            trace_rows.push(trace_rows.last().unwrap().clone());
        }
        TraceTable::init(transpose(trace_rows))
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
