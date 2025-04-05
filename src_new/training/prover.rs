// src/training/prover.rs

use crate::helper::{AC, FE, f64_to_felt, flatten_state_matrix, transpose};
use crate::training::air::{TrainingUpdateAir, TrainingUpdateInputs};
use winterfell::{
    AuxRandElements, CompositionPoly, CompositionPolyTrace, ConstraintCompositionCoefficients, DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde, PartitionOptions, ProofOptions, Prover, StarkDomain, TraceInfo, TracePolyTable, TraceTable
};
use winterfell::crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree};
use winterfell::math::FieldElement;
use winterfell::math::fields::f128::BaseElement as Felt;

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
        let state_width = AC * FE + AC;
        let mut state = flatten_state_matrix(&self.initial_w, &self.initial_b);
        let mut trace_rows = vec![state.clone()];
        let two = f64_to_felt(2.0);
        let fe = self.x.len();
        let ac = self.y.len();

        for _ in 1..self.trace_length {
            let mut new_state = state.clone();
            for j in 0..ac {
                let mut dot = f64_to_felt(0.0);
                for i in 0..fe {
                    let idx = j * fe + i;
                    dot = dot + state[idx] * self.x[i];
                }
                let bias_index = ac * fe + j;
                let pred = dot / self.precision + state[bias_index];
                let error = pred - self.y[j];
                for i in 0..fe {
                    let idx = j * fe + i;
                    let update_term = self.learning_rate * two * error * self.x[i] / self.precision;
                    new_state[idx] = state[idx] - update_term;
                }
                new_state[bias_index] = state[bias_index] - self.learning_rate * two * error;
            }
            trace_rows.push(new_state.clone());
            state = new_state;
        }
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

    fn get_pub_inputs(&self, _trace: &Self::Trace) -> TrainingUpdateInputs {
        let fe = self.x.len();
        let ac = self.y.len();
        let two = f64_to_felt(2.0);
        let mut state = flatten_state_matrix(&self.initial_w, &self.initial_b);

        for _ in 1..self.trace_length {
            let mut new_state = state.clone();
            for j in 0..ac {
                let mut dot = f64_to_felt(0.0);
                for i in 0..fe {
                    let idx = j * fe + i;
                    dot = dot + state[idx] * self.x[i];
                }
                let bias_index = ac * fe + j;
                let pred = dot / self.precision + state[bias_index];
                let error = pred - self.y[j];
                for i in 0..fe {
                    let idx = j * fe + i;
                    let update_term = self.learning_rate * two * error * self.x[i] / self.precision;
                    new_state[idx] = state[idx] - update_term;
                }
                new_state[bias_index] = state[bias_index] - self.learning_rate * two * error;
            }
            state = new_state;
        }

        TrainingUpdateInputs {
            initial: flatten_state_matrix(&self.initial_w, &self.initial_b),
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
