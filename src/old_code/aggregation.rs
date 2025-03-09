use winterfell::{
    math::{fields::f128::BaseElement as Felt, FieldElement},
    Air, AirContext, Assertion, ByteWriter, EvaluationFrame, FieldExtension, HashFunction,
    ProofOptions, Prover, Serializable, Trace, TraceInfo, TraceTable,
    TransitionConstraintDegree,
};
use std::time::Instant;
use hex; // Add hex = "0.4" to your Cargo.toml

/// Public inputs include the initial state, aggregated (final) state,
/// and the number of real multipliers.
#[derive(Clone)]
struct AggregationInputs {
    initial: Felt,
    aggregated: Felt,
    real_count: usize,
}

impl Serializable for AggregationInputs {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        target.write(self.initial);
        target.write(self.aggregated);
        target.write(Felt::new(self.real_count as u128));
    }
}

/// The AIR defines the constraint system. We use a two‑column trace:
///   - Column 0: “state” (the running product)
///   - Column 1: “multiplier” used in that transition
///
/// For each transition (row i, 0 ≤ i < L–1), we enforce:
///   state[i+1] = state[i] × multiplier[i].
struct AggregationAir {
    context: AirContext<Felt>,
    initial: Felt,
    aggregated: Felt,
}

impl Air for AggregationAir {
    type BaseField = Felt;
    type PublicInputs = AggregationInputs;

    fn new(trace_info: TraceInfo, pub_inputs: AggregationInputs, options: ProofOptions) -> Self {
        let degrees = vec![TransitionConstraintDegree::new(2)];
        let context = AirContext::new(trace_info, degrees, options);
        Self {
            context,
            initial: pub_inputs.initial,
            aggregated: pub_inputs.aggregated,
        }
    }

    // For each transition row (except the last) enforce:
    //   state_next – (state_current × multiplier_current) = 0.
    fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let state_current = frame.current()[0]; // Column 0: state
        let multiplier = frame.current()[1];      // Column 1: multiplier
        let state_next = frame.next()[0];          // Next row, column 0: next state
        result[0] = state_current * multiplier - state_next;
    }

    // Assert initial state and final (aggregated) state.
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

/// The prover builds a two‑column trace.
/// - Column 0 (“state”): computed as a running product starting at the initial value.
/// - Column 1 (“multiplier”): contains the real multipliers for the first transitions,
///   then dummy multipliers (1) for padding.
/// The trace length is padded to the next power of two.
struct AggregationProver {
    options: ProofOptions,
    // The real multipliers (e.g. client commitments)
    real_multipliers: Vec<Felt>,
    trace_length: usize,
}

impl AggregationProver {
    /// Creates a new prover.
    fn new(options: ProofOptions, real_multipliers: Vec<Felt>) -> Self {
        // Real trace length: number of real transitions = real_multipliers.len() + 1.
        let real_trace_length = real_multipliers.len() + 1;
        // Pad the trace length to the next power of two.
        let trace_length = real_trace_length.next_power_of_two();
        Self {
            options,
            real_multipliers,
            trace_length,
        }
    }

    /// Build a two‑column trace.
    /// Column 0 ("state"): state[0] = initial, state[i+1] = state[i] × multiplier[i].
    /// Column 1 ("multiplier"):
    ///   - For 0 ≤ i < real_multipliers.len(), use the real multiplier.
    ///   - For dummy transitions, use 1.
    /// Finally, if needed, pad the multiplier column with one extra dummy value so that both columns have equal length.
    fn build_trace(initial: Felt, real_multipliers: &[Felt], trace_length: usize) -> TraceTable<Felt> {
        let mut state_trace = Vec::with_capacity(trace_length);
        let mut multiplier_trace = Vec::with_capacity(trace_length);
        state_trace.push(initial);
        // Add real transitions.
        for &m in real_multipliers {
            let prev = *state_trace.last().unwrap();
            let next = prev * m;
            state_trace.push(next);
            multiplier_trace.push(m);
        }
        // Add dummy transitions until state_trace has the target length.
        while state_trace.len() < trace_length {
            let prev = *state_trace.last().unwrap();
            let next = prev; // using multiplier 1 leaves state unchanged.
            state_trace.push(next);
            multiplier_trace.push(Felt::new(1));
        }
        // If for any reason the multiplier column is one entry short, pad it.
        if multiplier_trace.len() < state_trace.len() {
            multiplier_trace.push(Felt::new(1));
        }
        TraceTable::init(vec![state_trace, multiplier_trace])
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
            real_count: self.real_multipliers.len(),
        }
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }
}

//
// Main: Demonstrate proof generation, printing, and verification.
//
fn main() {
    // Example real multipliers (e.g., client commitments). For instance, [2, 3, 5, 7] yields 210.
    let real_multipliers = vec![
        Felt::new(2),
        Felt::new(3),
        Felt::new(5),
        Felt::new(7),
    ];
    let initial = Felt::new(1);
    // Set STARK proof options.
    // Increase the blowup factor from 4 to 16 to ensure the extended domain is large enough.
    let proof_options = ProofOptions::new(
        40,                // security level in bits
        16,                // blowup factor
        21,                // grinding factor
        HashFunction::Blake3_256,
        FieldExtension::None,
        8,                 // fri_layout parameter (example)
        64,                // fri_max_degree parameter (example)
    );
    // Create the prover (this computes the padded trace length).
    let prover = AggregationProver::new(proof_options, real_multipliers);
    // Build the trace.
    let trace = AggregationProver::build_trace(initial, &prover.real_multipliers, prover.trace_length);
    let aggregated = trace.get(0, trace.length() - 1);
    println!("Aggregated commitment from trace: {:?}", aggregated);
    // Generate the STARK proof.
    let start_time = Instant::now();
    let proof = prover.prove(trace).expect("Proof generation failed");
    println!("Proof generated in {} ms", start_time.elapsed().as_millis());

    // Serialize the proof to bytes and print as a hex string.
    let proof_bytes = proof.to_bytes();
    let proof_hex = hex::encode(&proof_bytes);
    println!("Proof (hex): {}", proof_hex);

    // Serialize the public inputs and print as a hex string.
    let mut pub_inputs_bytes = Vec::new();
    let pub_inputs = AggregationInputs {
        initial,
        aggregated,
        real_count: prover.real_multipliers.len(),
    };
    pub_inputs.write_into(&mut pub_inputs_bytes);
    let pub_inputs_hex = hex::encode(&pub_inputs_bytes);
    println!("Public inputs (hex): {}", pub_inputs_hex);

    // Verify the proof.
    let verify_start = Instant::now();
    match winterfell::verify::<AggregationAir>(proof, pub_inputs) {
        Ok(_) => println!("Proof verified in {} μs", verify_start.elapsed().as_micros()),
        Err(e) => println!("Verification failed: {}", e),
    }
}
