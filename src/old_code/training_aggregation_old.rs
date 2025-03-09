use winterfell::{
    math::{fields::f128::BaseElement as Felt, FieldElement},
    Air, AirContext, Assertion, ByteWriter, EvaluationFrame, FieldExtension, HashFunction,
    ProofOptions, Prover, Serializable, Trace, TraceInfo, TraceTable,
    TransitionConstraintDegree,
};
use std::time::Instant;

// ------------------------
// MODULE: Training Update Verification (Client Side)
// ------------------------
mod training {
    use super::*;
    
    /// Public inputs for the training update circuit.
    /// - `initial`: initial weight (or model parameter),
    /// - `final_weight`: weight after applying all gradients,
    /// - `steps`: number of training steps (gradients).
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
    
    /// The AIR for training update verification.
    /// We model the training update as:
    ///    weight[i+1] = weight[i] - gradient[i]
    /// using a two-column trace:
    ///   - Column 0: running weight
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
            let degrees = vec![TransitionConstraintDegree::new(1)]; // linear constraint
            let context = AirContext::new(trace_info, degrees, options);
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
            // Enforce: weight[i+1] = weight[i] - gradient[i]
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
    
    /// The TrainingProver builds the trace for a training update.
    /// Given an initial weight and a sequence of gradients, it computes:
    ///   weight[0] = initial, then weight[i+1] = weight[i] - gradient[i].
    /// The trace is padded to the next power of two, with a minimum of 8 steps.
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
            // Add real transitions.
            for &g in &self.gradients {
                let prev = *weight_trace.last().unwrap();
                let next = prev - g;
                weight_trace.push(next);
                gradient_trace.push(g);
            }
            // Add dummy transitions until weight_trace has the target length.
            while weight_trace.len() < self.trace_length {
                let prev = *weight_trace.last().unwrap();
                weight_trace.push(prev);
                gradient_trace.push(Felt::new(0));
            }
            // Ensure both registers have the same length.
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
    
        fn get_pub_inputs(&self, trace: &Self::Trace) -> TrainingInputs {
            let last = trace.length() - 1;
            TrainingInputs {
                initial: trace.get(0, 0),
                final_weight: trace.get(0, last),
                steps: self.gradients.len(),
            }
        }
    
        fn options(&self) -> &ProofOptions {
            &self.options
        }
    }
}

// ------------------------
// MODULE: Aggregation Verification (Server Side)
// ------------------------
mod aggregation {
    use super::*;
    
    /// Public inputs for the aggregation circuit.
    /// - `initial`: starting sum (e.g., 0),
    /// - `aggregated`: final aggregated value (e.g., sum of client updates),
    /// - `count`: number of client updates aggregated.
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
    
    /// The AggregationAir defines the constraint system for verifying aggregation.
    /// We use a two‑column trace:
    ///   - Column 0: running sum
    ///   - Column 1: the client update added at that step.
    /// The constraint is: sum[i+1] = sum[i] + update[i], i.e.,
    /// sum[i] + update[i] − sum[i+1] = 0.
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
            let context = AirContext::new(trace_info, degrees, options);
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
            let sum_current = frame.current()[0];   // running sum
            let update = frame.current()[1];          // update added in this step
            let sum_next = frame.next()[0];           // next running sum
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
    
    /// The AggregationProver builds a two‑column trace representing the running sum of client updates.
    /// Given a vector of client updates and an initial value (e.g., 0),
    /// it produces a trace:
    ///   sum[0] = initial, and for each update:
    ///   sum[i+1] = sum[i] + update[i]
    /// Column 1 holds the update values.
    /// The trace is padded to the next power of two, with a minimum of 8 steps.
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
    }
}

use training::TrainingProver;
use aggregation::{AggregationAir, AggregationProver};

//
// MAIN: Simulate a FedXgbBagging round using zkSTARKs for both local training and aggregation.
//
// In this simulation:
//  - Multiple clients perform local training and generate proofs for their updates.
//  - The server collects the clients’ final weights, (optionally) verifies the individual proofs,
//    and then aggregates the weights. An aggregation zkSTARK is generated to prove that the sum is correct.
//  - The proofs and public inputs are serialized as hexadecimal strings for external sharing and verification.
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
        40,    // security level
        16,    // blowup factor
        21,    // grinding factor
        HashFunction::Blake3_256,
        FieldExtension::None,
        8,     // fri_layout parameter (example)
        64,    // fri_max_degree parameter (example)
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
        // In a real system, the client sends the final weight and proof to the server.
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
    
    // Verify the aggregation proof.
    match winterfell::verify::<AggregationAir>(agg_proof, agg_pub_inputs) {
        Ok(_) => println!("Aggregation proof verified successfully."),
        Err(e) => println!("Aggregation proof verification failed: {}", e),
    }
}
