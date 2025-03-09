use winterfell::{
    math::{fields::f128::BaseElement as Felt, FieldElement}, Air, AirContext, Assertion, ByteWriter, EvaluationFrame, FieldExtension, HashFunction, ProofOptions, Prover, Serializable, StarkProof, Trace, TraceInfo, TraceTable, TransitionConstraintDegree
};
use std::{hash::Hash, result, time::Instant, vec};

// CONSTANTS
// ===============================
const ALPHA: u64 = 3;
const INV_ALPHA: u128 = 226854911280625642308916371969163307691;
const FORTY_TWO: Felt = Felt::new(42);

// MAIN FUNCTION
// ===============================

pub fn main() {
    let n = 1024 * 1024;
    let seed= Felt::new(5);

    // compute result
    let now = Instant::now();
    let result = vdf(seed, n);
    println!("VDF computation took {} ms", now.elapsed().as_millis());

    // specify parameters for the STARK protocol and create a prover
    let stark_params = ProofOptions::new(
        40,
        4,
        21,
        HashFunction::Blake3_256,
        FieldExtension::None,
        8,
        64,
    );

    // instatiate the prover
    let prover = VdfProver::new(stark_params);

    // build execution trace
    let now = Instant::now();
    let trace = VdfProver::build_trace(seed, n);
    println!("Building trace took {} ms", now.elapsed().as_millis());
    assert_eq!(result, trace.get(0, n - 1));

    // generate proof
    let now = Instant::now();
    let proof = prover.prove(trace).unwrap();
    println!("Proof generation took {} ms", now.elapsed().as_millis());

    // serialize proof and check security level
    let proof_bytes = proof.to_bytes();
    println!("Proof size: {:.1} KB", proof_bytes.len() as f64 / 1024f64);
    println!("Security level: {} bits", proof.security_level(true));

    // deserialize proof
    let parsed_proof = StarkProof::from_bytes(&proof_bytes).unwrap();
    assert_eq!(proof, parsed_proof);

    //initalize public inputs
    let pub_inputs = VdfInputs {
        seed,
        result,
    };

    // verify proof
    let now = Instant::now();
    match winterfell::verify::<VdfAir>(proof, pub_inputs) {
        Ok(_) => println!("Proof verified in {:.} ms", now.elapsed().as_millis()),
        Err(msg) => println!("Verification failed: {}", msg),
    }
}

// VDF FUNCTION
// ===============================

fn vdf(seed: Felt, n: usize) -> Felt {
    let mut state = seed;
    for _ in 0..(n - 1) {
        state = (state- FORTY_TWO).exp(INV_ALPHA);
    }
    state
}

// PUBLIC INPUTS
// ===============================

#[derive(Clone)]
struct VdfInputs {
    seed: Felt,
    result: Felt,
}

impl Serializable for VdfInputs {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        target.write(self.seed);
        target.write(self.result);
    }
}

// VDF AIR
// ===============================
struct VdfAir {
    context: AirContext<Felt>,
    seed: Felt,
    result: Felt,
}

impl Air for VdfAir {
    type BaseField = Felt;
    type PublicInputs = VdfInputs;

    fn new(trace_info: TraceInfo, pub_inputs: VdfInputs, options: ProofOptions) -> Self {
        let degrees = vec![TransitionConstraintDegree::new(3)];
        let context = AirContext::new(trace_info, degrees, options);
        Self {
            context,
            seed: pub_inputs.seed,
            result: pub_inputs.result,
        }
    }

    fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    )
    {
        let current_state = frame.current()[0];
        let next_state = frame.next()[0];

        result[0] = current_state - (next_state.exp(ALPHA.into())) - FORTY_TWO.into();
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let last_step = self.trace_length() - 1;
        vec![
            Assertion::single(0, 0, self.seed),
            Assertion::single(0, last_step, self.result),
        ]
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}


// PROVER
// ===============================

struct VdfProver {
    options: ProofOptions,
}

impl VdfProver {
    pub fn new(options: ProofOptions) -> Self {
        Self { options }
    }

    pub fn build_trace(seed: Felt, n: usize) -> TraceTable<Felt> {
        let mut trace = Vec::with_capacity(n);

        let mut state = seed;
        trace.push(state);

        for _ in 0..(n - 1) {
            state = (state - FORTY_TWO).exp(INV_ALPHA);
            trace.push(state);
        }

        TraceTable::init(vec![trace])
    }
}

impl Prover for VdfProver {
    type BaseField = Felt;
    type Air = VdfAir;
    type Trace = TraceTable<Felt>;

    fn get_pub_inputs(&self, trace: &Self::Trace) -> VdfInputs {
        let last_step = trace.length() - 1;
        VdfInputs {
            seed: trace.get(0, 0),
            result: trace.get(0, last_step),
        }
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }
}