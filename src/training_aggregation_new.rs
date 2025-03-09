use std::time::Instant;
use winterfell::{
    Air, AirContext, Assertion, ByteWriter, EvaluationFrame, FieldExtension, ProofOptions, Prover,
    Trace, TraceInfo, TraceTable, TransitionConstraintDegree, AcceptableOptions,
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
    
    // Implement ToElements for TrainingInputs.
    impl ToElements<Felt> for TrainingInputs {
        fn to_elements(&self) -> Vec<Felt> {
            vec![self.initial, self.final_weight, Felt::new(self.steps as u128)]
        }
    }
    
    /// The AIR for training update verification.
    /// Models the update as: weight[i+1] = weight[i] - gradient[i]
    /// with a two‑column trace:
    ///   - Column 0: running weight,
    ///   - Column 1: gradient applied at that step.
    pub struct TrainingAir {
        context: AirContext<Felt>,
        initial: Felt,
        final_weight: Felt,
    }
    
    impl Air for TrainingAir {
        type BaseField = Felt;
        type PublicInputs = TrainingInputs;
    
        fn new(trace_info: TraceInfo, pub_inputs: TrainingInputs, options: ProofOptions) -> Self {
            let degrees = vec![TransitionConstraintDegree::new(1)]; // one linear constraint
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
    
    /// The TrainingProver builds the execution trace for a training update.
    /// Given an initial weight and a list of gradients, it computes:
    ///   weight[0] = initial, then weight[i+1] = weight[i] - gradient[i].
    /// The trace is padded to the next power of two (minimum 8 steps).
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
            // Compute the real transitions.
            for &g in &self.gradients {
                let prev = *weight_trace.last().unwrap();
                let next = prev - g;
                weight_trace.push(next);
                gradient_trace.push(g);
            }
            // Pad the trace to the target length.
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
        type TraceLde<E: FieldElement<BaseField = Self::BaseField>> = DefaultTraceLde<E, Self::HashFn, Self::VC>;
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

// ------------------------
// MODULE: Aggregation Verification (Server Side)
// ------------------------
mod aggregation {
    use super::*;
    
    /// Public inputs for the aggregation circuit.
    #[derive(Clone)]
    pub struct AggregationInputs {
        pub initial: Felt,
        pub aggregated: Felt,
        pub count: usize,
    }
    
    impl Serializable for AggregationInputs {
        fn write_into<W: ByteWriter>(&self, target: &mut W) {
            target.write(self.initial);
            target.write(self.aggregated);
            target.write(Felt::new(self.count as u128));
        }
    }
    
    // Implement ToElements for AggregationInputs.
    impl ToElements<Felt> for AggregationInputs {
        fn to_elements(&self) -> Vec<Felt> {
            vec![self.initial, self.aggregated, Felt::new(self.count as u128)]
        }
    }
    
    /// The AggregationAir defines the constraint system for verifying aggregation.
    /// Uses a two‑column trace where:
    ///   - Column 0: running sum,
    ///   - Column 1: update added at that step.
    pub struct AggregationAir {
        context: AirContext<Felt>,
        initial: Felt,
        aggregated: Felt,
    }
    
    impl Air for AggregationAir {
        type BaseField = Felt;
        type PublicInputs = AggregationInputs;
    
        fn new(trace_info: TraceInfo, pub_inputs: AggregationInputs, options: ProofOptions) -> Self {
            let degrees = vec![TransitionConstraintDegree::new(1)];
            let context = AirContext::new(trace_info, degrees, 2, options);
            Self {
                context,
                initial: pub_inputs.initial,
                aggregated: pub_inputs.aggregated,
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
    
            let sum_current = frame.current()[0];
            let update = frame.current()[1];
            let sum_next = frame.next()[0];
            result[0] = sum_current + update - sum_next;
        }
    
        fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
            let trace_len = self.trace_length();
            vec![
                Assertion::single(0, 0, self.initial),
                Assertion::single(0, trace_len - 1, self.aggregated),
            ]
        }
    
        fn context(&self) -> &AirContext<Self::BaseField> {
            &self.context
        }
    }
    
    /// The AggregationProver builds the trace representing the running sum.
    /// Given an initial value and a list of updates, it computes:
    ///   sum[0] = initial and sum[i+1] = sum[i] + update[i]
    /// (with padding to a power-of-two length, minimum 8 steps).
    pub struct AggregationProver {
        pub options: ProofOptions,
        pub updates: Vec<Felt>,
        pub trace_length: usize,
    }
    
    impl AggregationProver {
        pub fn new(options: ProofOptions, updates: Vec<Felt>) -> Self {
            let real_length = updates.len() + 1;
            let mut trace_length = real_length.next_power_of_two();
            if trace_length < 8 {
                trace_length = 8;
            }
            Self { options, updates, trace_length }
        }
    
        pub fn build_trace(initial: Felt, updates: &[Felt], trace_length: usize) -> TraceTable<Felt> {
            let mut sum_trace = Vec::with_capacity(trace_length);
            let mut update_trace = Vec::with_capacity(trace_length);
            sum_trace.push(initial);
            for &u in updates {
                let prev = *sum_trace.last().unwrap();
                let next = prev + u;
                sum_trace.push(next);
                update_trace.push(u);
            }
            while sum_trace.len() < trace_length {
                let prev = *sum_trace.last().unwrap();
                sum_trace.push(prev);
                update_trace.push(Felt::new(0));
            }
            if update_trace.len() < sum_trace.len() {
                update_trace.push(Felt::new(0));
            }
            TraceTable::init(vec![sum_trace, update_trace])
        }
    }
    
    impl Prover for AggregationProver {
        type BaseField = Felt;
        type Air = AggregationAir;
        type Trace = TraceTable<Felt>;
        type HashFn = Blake3_256<Felt>;
        type VC = MerkleTree<Self::HashFn>;
        type RandomCoin = DefaultRandomCoin<Self::HashFn>;
        type TraceLde<E: FieldElement<BaseField = Self::BaseField>> = DefaultTraceLde<E, Self::HashFn, Self::VC>;
        type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
            DefaultConstraintEvaluator<'a, Self::Air, E>;
        type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
            DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;
    
        fn get_pub_inputs(&self, trace: &Self::Trace) -> AggregationInputs {
            let last = trace.length() - 1;
            AggregationInputs {
                initial: trace.get(0, 0),
                aggregated: trace.get(0, last),
                count: self.updates.len(),
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

use training::TrainingProver;
use aggregation::{AggregationAir, AggregationProver};

fn main() {
    // ----------------------- Client Side (Training Updates) -----------------------
    // Simulate three clients.
    let client_initial_weights = vec![Felt::new(100), Felt::new(120), Felt::new(90)];
    // Each client applies its local gradients.
    let client_gradients = vec![
        vec![Felt::new(5), Felt::new(3)],  // Client 1: 100 - 5 - 3 = 92
        vec![Felt::new(10), Felt::new(5)], // Client 2: 120 - 10 - 5 = 105
        vec![Felt::new(8), Felt::new(2)],  // Client 3: 90 - 8 - 2 = 80
    ];
    
    let mut client_final_weights = Vec::new();
    let mut client_training_proofs = Vec::new();
    let client_proof_options = ProofOptions::new(
        40,                       // security level
        16,                       // blowup factor
        21,                       // grinding factor
        FieldExtension::None,     // field extension (if that's what your version expects)
        16,                       // fri_max_degree parameter (changed from 64 to 16)
        7,                        // fri_layout parameter
        BatchingMethod::Algebraic, // first batching method
        BatchingMethod::Algebraic  // second batching method
    );
            
    println!("--- Client Training Updates ---");
    for (i, gradients) in client_gradients.iter().enumerate() {
        let initial = client_initial_weights[i];
        let training_prover = TrainingProver::new(client_proof_options.clone(), initial, gradients.clone());
        let trace = training_prover.build_trace();
        let pub_inputs = training_prover.get_pub_inputs(&trace);
        let start = Instant::now();
        let proof = training_prover.prove(trace).expect("Training proof generation failed");
        let duration = start.elapsed().as_millis();
        println!("Client {}:", i + 1);
        println!("  Initial weight: {:?}", pub_inputs.initial);
        println!("  Final weight:   {:?}", pub_inputs.final_weight);
        println!("  Proof generated in {} ms", duration);
        client_final_weights.push(pub_inputs.final_weight);
        client_training_proofs.push(proof);
    }
    
    // ----------------------- Server Side (Aggregation) -----------------------
    // The server aggregates the client final weights.
    let aggregated_update = client_final_weights.iter().fold(Felt::new(0), |acc, &w| acc + w);
    
    println!("\n--- Server Aggregation ---");
    println!("Client final weights: {:?}", client_final_weights);
    println!("Aggregated update (sum): {:?}", aggregated_update);
    
    let aggregation_prover = AggregationProver::new(
        client_proof_options.clone(),
        client_final_weights.clone(),
    );
    let agg_trace = AggregationProver::build_trace(Felt::new(0), &aggregation_prover.updates, aggregation_prover.trace_length);
    let agg_pub_inputs = aggregation_prover.get_pub_inputs(&agg_trace);
    
    let start = Instant::now();
    let agg_proof = aggregation_prover.prove(agg_trace).expect("Aggregation proof generation failed");
    println!("Aggregation proof generated in {} ms", start.elapsed().as_millis());
    
    // Serialize aggregation proof and public inputs as hex strings for sharing.
    let agg_proof_bytes = agg_proof.to_bytes();
    let agg_proof_hex = hex::encode(&agg_proof_bytes);
    let mut agg_pub_bytes = Vec::new();
    agg_pub_inputs.write_into(&mut agg_pub_bytes);
    let agg_pub_hex = hex::encode(&agg_pub_bytes);
    
    println!("Aggregation proof (hex): {}", agg_proof_hex);
    println!("Aggregation public inputs (hex): {}", agg_pub_hex);

    let acceptable_options = AcceptableOptions::OptionSet(vec![client_proof_options.clone()]);

    
    // Verify the aggregation proof.
    match winterfell::verify::<AggregationAir, Blake3_256<Felt>, DefaultRandomCoin<Blake3_256<Felt>>, MerkleTree<Blake3_256<Felt>>>(
        agg_proof,
        agg_pub_inputs,
        &acceptable_options,
    ) {
        Ok(_) => println!("Aggregation proof verified successfully."),
        Err(e) => println!("Aggregation proof verification failed: {}", e),
    }}
