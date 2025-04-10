use crate::helper::{f64_to_felt, AC, FE};
use winterfell::{
    Air, AirContext, Assertion, EvaluationFrame, ProofOptions, TraceInfo,
    TransitionConstraintDegree,
};
use winter_utils::Serializable;
use winterfell::math::{FieldElement, ToElements};
use winterfell::math::fields::f128::BaseElement as Felt;

/// Public inputs for the aggregation circuit.
#[derive(Clone)]
pub struct GlobalUpdateInputs {
    pub global_w: Vec<Vec<Felt>>,    // initial global weights [AC x FE]
    pub global_b: Vec<Felt>,         // initial global biases [AC]
    pub new_global_w: Vec<Vec<Felt>>, // aggregated (updated) global weights [AC x FE]
    pub new_global_b: Vec<Felt>,      // aggregated (updated) global biases [AC]
    pub k: Felt,                     // scaling factor (number of clients)
    pub digest: Felt,                // MiMC hash digest (can be computed later)
    pub steps: usize,                // number of update steps (uns‑padded rows)
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
        target.write(winterfell::math::fields::f128::BaseElement::new(self.steps as u128));
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
        elems.push(winterfell::math::fields::f128::BaseElement::new(self.steps as u128));
        elems
    }
}

/// AIR for the aggregation circuit.
pub struct GlobalUpdateAir {
    pub context: AirContext<Felt>,
    pub pub_inputs: GlobalUpdateInputs,
}

impl GlobalUpdateAir {
    /// Get the public initial state S₀ (flattened global weights then biases).
    fn get_public_initial_state(&self) -> Vec<Felt> {
        let mut state = Vec::new();
        for i in 0..AC {
            for j in 0..FE {
                state.push(self.pub_inputs.global_w[i][j]);
            }
        }
        for i in 0..AC {
            state.push(self.pub_inputs.global_b[i]);
        }
        state
    }

    /// Get the aggregated state.
    fn get_public_new_state(&self) -> Vec<Felt> {
        let mut state = Vec::new();
        for i in 0..AC {
            for j in 0..FE {
                state.push(self.pub_inputs.new_global_w[i][j]);
            }
        }
        for i in 0..AC {
            state.push(self.pub_inputs.new_global_b[i]);
        }
        state
    }
}

impl Air for GlobalUpdateAir {
    type BaseField = Felt;
    type PublicInputs = GlobalUpdateInputs;

    fn new(trace_info: TraceInfo, pub_inputs: GlobalUpdateInputs, options: ProofOptions) -> Self {
        let width = (AC * FE + AC) * 2;
        let degrees = vec![TransitionConstraintDegree::new(1); AC * FE + AC];
        let context = AirContext::new(trace_info, degrees, width, options);
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
        for i in 0..d {
            let current_state = frame.current()[i];
            let next_state = frame.next()[i];
            let update_raw = frame.next()[i + d];
            result[i] = c * next_state - c * current_state - update_raw;
        }
        for i in d..result.len() {
            result[i] = E::ZERO;
        }
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let d = AC * FE + AC;
        let new_state = self.get_public_new_state();
        let last_row = self.pub_inputs.steps - 1;
        let mut assertions = Vec::new();
        for i in 0..d {
            assertions.push(Assertion::single(i, last_row, new_state[i]));
        }
        for i in d..(2 * d) {
            assertions.push(Assertion::single(i, last_row, Self::BaseField::ZERO));
        }
        assertions
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}
