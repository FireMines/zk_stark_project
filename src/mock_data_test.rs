use std::time::Instant;
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

// ------------------------
// MODULE: Training Update Verification (Client Side)
// ------------------------
mod training {
    use super::*;

    /// Public inputs for the training update circuit.
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
    
    /// The AIR for training update verification.
    /// It models the update: weight[i+1] = weight[i] - gradient[i] using a two‑column trace.
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
    
        fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            frame: &EvaluationFrame<E>,
            _periodic_values: &[E],
            result: &mut [E],
        ) {
            debug_assert_eq!(frame.current().len(), 2);
            debug_assert_eq!(frame.next().len(), 2);
            let weight_current = frame.current()[0];
            let gradient = frame.current()[1];
            let weight_next = frame.next()[0];
            result[0] = weight_current - gradient - weight_next;
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
    
    /// The TrainingProver builds the execution trace.
    /// It computes weight[0] = initial and weight[i+1] = weight[i] - gradient[i],
    /// padding the trace to a power‑of‑two length (minimum 8 steps).
    pub struct TrainingProver {
        pub options: ProofOptions,
        pub gradients: Vec<Felt>,
        pub trace_length: usize,
        pub initial: Felt,
    }
    
    impl TrainingProver {
        pub fn new(options: ProofOptions, initial: Felt, gradients: Vec<Felt>) -> Self {
            let real_length = gradients.len() + 1;
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

// For this test, we focus on the training update.
use training::TrainingProver;

/// Test the fake data.
/// Fake model parameters: {"weight_1": 0.5, "weight_2": -0.3}
/// We scale these by 100: weight1 = 50, weight2 = 30, so initial = 50 - 30 = 20.
/// Instead of a no-op update, we use a minimal update [1] so that final becomes 20 - 1 = 19.
fn test_fake_data() {
    // Scale fake parameters.
    let weight1 = Felt::new(50);
    let weight2 = Felt::new(30);
    let fake_model_value = weight1 - weight2; // 20

    // Use a minimal non-zero update: [1]
    let gradients = vec![Felt::new(1)]; // Expected final weight: 20 - 1 = 19

    let prover_options = ProofOptions::new(
        40, 16, 21, FieldExtension::None, 16, 7,
        BatchingMethod::Algebraic, BatchingMethod::Algebraic,
    );
    let training_prover = TrainingProver::new(prover_options, fake_model_value, gradients);
    let trace = training_prover.build_trace();
    let pub_inputs = training_prover.get_pub_inputs(&trace);

    println!("Fake data test:");
    println!("  Initial weight: {:?}", pub_inputs.initial);
    println!("  Final weight:   {:?}", pub_inputs.final_weight);
    // Expected final weight is 20 - 1 = 19.
    let expected_final = fake_model_value - Felt::new(1);
    assert_eq!(pub_inputs.final_weight, expected_final);
    
    let proof = training_prover.prove(trace).expect("Proof generation failed");
    println!("Fake model update proof (hex): {}", hex::encode(proof.to_bytes()));
}

fn main() {
    test_fake_data();
}
