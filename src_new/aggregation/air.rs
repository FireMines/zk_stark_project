// src/aggregation/air.rs

use crate::helper::{AC, FE};
use winterfell::{
    Air, AirContext, Assertion, EvaluationFrame, TraceInfo, TransitionConstraintDegree, ProofOptions,
};
use winter_utils::Serializable;
use winterfell::math::{FieldElement, ToElements};
use winterfell::math::fields::f128::BaseElement as Felt;

#[derive(Clone)]
pub struct GlobalUpdateInputs {
    pub global_w: Vec<Vec<Felt>>,    // initial global weights [AC x FE]
    pub global_b: Vec<Felt>,         // initial global biases [AC]
    pub new_global_w: Vec<Vec<Felt>>, // aggregated new weights [AC x FE]
    pub new_global_b: Vec<Felt>,      // aggregated new biases [AC]
    pub k: Felt,                     // scaling factor
    pub digest: Felt,                // MiMC hash digest of new model
}

impl Serializable for GlobalUpdateInputs {
    fn write_into<W: winterfell::ByteWriter>(&self, target: &mut W) {
        for i in 0..AC {
            for j in 0..FE {
                target.write(self.global_w[i][j]);
            }
        }
        for i in 0..AC { target.write(self.global_b[i]); }
        for i in 0..AC {
            for j in 0..FE {
                target.write(self.new_global_w[i][j]);
            }
        }
        for i in 0..AC { target.write(self.new_global_b[i]); }
        target.write(self.k);
        target.write(self.digest);
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
        for i in 0..AC { elems.push(self.global_b[i]); }
        for i in 0..AC {
            for j in 0..FE {
                elems.push(self.new_global_w[i][j]);
            }
        }
        for i in 0..AC { elems.push(self.new_global_b[i]); }
        elems.push(self.k);
        elems.push(self.digest);
        elems
    }
}

pub struct GlobalUpdateAir {
    pub context: AirContext<Felt>,
    pub pub_inputs: GlobalUpdateInputs,
}

impl GlobalUpdateAir {
    fn get_public_initial_state(&self) -> Vec<Felt> {
        let mut state = Vec::new();
        for i in 0..AC {
            for j in 0..FE { state.push(self.pub_inputs.global_w[i][j]); }
        }
        for i in 0..AC { state.push(self.pub_inputs.global_b[i]); }
        state
    }

    fn get_public_new_state(&self) -> Vec<Felt> {
        let mut state = Vec::new();
        for i in 0..AC {
            for j in 0..FE { state.push(self.pub_inputs.new_global_w[i][j]); }
        }
        for i in 0..AC { state.push(self.pub_inputs.new_global_b[i]); }
        state
    }
}

impl Air for GlobalUpdateAir {
    type BaseField = Felt;
    type PublicInputs = GlobalUpdateInputs;

    fn new(trace_info: TraceInfo, pub_inputs: GlobalUpdateInputs, options: ProofOptions) -> Self {
        let state_width = AC * FE + AC;
        let degrees = vec![TransitionConstraintDegree::new(1); state_width];
        let context = AirContext::new(trace_info, degrees, state_width, options);
        Self { context, pub_inputs }
    }

    fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField> + From<Felt>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let n_minus_one = E::from(Felt::new((self.trace_length() - 1) as u128)).inv();
        let public_initial = self.get_public_initial_state();
        let public_final = self.get_public_new_state();
        let expected_deltas: Vec<E> = public_initial
            .iter()
            .zip(public_final.iter())
            .map(|(init, fin)| E::from(*fin - *init) * n_minus_one)
            .collect();
        for i in 0..result.len() {
            let actual_delta = frame.next()[i] - frame.current()[i];
            result[i] = actual_delta - expected_deltas[i];
        }
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let public_final = self.get_public_new_state();
        (0..public_final.len())
            .map(|i| Assertion::single(i, self.trace_length() - 1, public_final[i]))
            .collect()
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}
