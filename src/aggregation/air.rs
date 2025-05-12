// aggregation_seperated/aggregation/air.rs

use crate::helper::{AC, FE};
use winterfell::{
    Air, AirContext, Assertion, EvaluationFrame, ProofOptions, TraceInfo,
    TransitionConstraintDegree,
};
use winter_utils::Serializable;
use winterfell::math::{FieldElement, ToElements};
use winterfell::math::fields::f128::BaseElement as Felt;

/// The public inputs are masked old and new states plus a digest.
#[derive(Clone)]
pub struct GlobalUpdateInputs {
    // masked old state
    pub global_w: Vec<Vec<Felt>>,
    pub global_b: Vec<Felt>,

    // masked new state
    pub new_global_w: Vec<Vec<Felt>>,
    pub new_global_b: Vec<Felt>,

    // scaling factor (masked or unmasked? Usually unmasked, e.g. 8e6.)
    pub k: Felt,

    // a MiMC digest
    pub digest: Felt,

    // number of steps in the unpadded trace
    pub steps: usize,
}

impl Serializable for GlobalUpdateInputs {
    fn write_into<W: winterfell::ByteWriter>(&self, target: &mut W) {
        for i in 0..AC {
            for j in 0..FE {
                target.write(self.global_w[i][j]);
            }
        }
        for i in 0..AC {
            target.write(self.global_b[i]);
        }
        for i in 0..AC {
            for j in 0..FE {
                target.write(self.new_global_w[i][j]);
            }
        }
        for i in 0..AC {
            target.write(self.new_global_b[i]);
        }
        target.write(self.k);
        target.write(self.digest);
        target.write(Felt::new(self.steps as u128));
    }
}

impl ToElements<Felt> for GlobalUpdateInputs {
    fn to_elements(&self) -> Vec<Felt> {
        let mut elems = Vec::new();
        for i in 0..AC {
            for j in 0..FE {
                elems.push(self.global_w[i][j]);
            }
        }
        for i in 0..AC {
            elems.push(self.global_b[i]);
        }
        for i in 0..AC {
            for j in 0..FE {
                elems.push(self.new_global_w[i][j]);
            }
        }
        for i in 0..AC {
            elems.push(self.new_global_b[i]);
        }
        elems.push(self.k);
        elems.push(self.digest);
        elems.push(Felt::new(self.steps as u128));
        elems
    }
}

/// The aggregator AIR verifying the masked iterative update.
pub struct GlobalUpdateAir {
    pub context: AirContext<Felt>,
    pub pub_inputs: GlobalUpdateInputs,
}

impl Air for GlobalUpdateAir {
    type BaseField = Felt;
    type PublicInputs = GlobalUpdateInputs;

    fn new(trace_info: TraceInfo, pub_inputs: GlobalUpdateInputs, options: ProofOptions) -> Self {
        let d = AC * FE + AC;
        let degrees = vec![TransitionConstraintDegree::new(1); d];
        // total width = 2*d
        let context = AirContext::new(trace_info, degrees, d * 2, options);
        Self { context, pub_inputs }
    }

    fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField> + From<Felt>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let d = AC * FE + AC;
        let c = E::from(self.pub_inputs.k);
        // for i in 0..d, constraint is c*(S_{next} - S_{curr}) - U = 0
        for i in 0..d {
            let curr = frame.current()[i];
            let next = frame.next()[i];
            let update = frame.next()[i + d];
            result[i] = c * next - c * curr - update;
        }
        for i in d..result.len() {
            result[i] = E::ZERO;
        }
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let d = AC * FE + AC;
        let final_state_flat = {
            let mut buf = Vec::with_capacity(d);
            for i in 0..AC {
                for j in 0..FE {
                    buf.push(self.pub_inputs.new_global_w[i][j]);
                }
            }
            for i in 0..AC {
                buf.push(self.pub_inputs.new_global_b[i]);
            }
            buf
        };
        let last_row = self.pub_inputs.steps - 1;

        let mut assertions = Vec::new();
        // S-part in last row must match masked final. 
        for i in 0..d {
            assertions.push(Assertion::single(i, last_row, final_state_flat[i]));
        }
        // update part in last row must be zero.
        for i in d..(2*d) {
            assertions.push(Assertion::single(i, last_row, Felt::ZERO));
        }
        assertions
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}
