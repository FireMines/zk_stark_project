// src/training/air.rs

use crate::helper::f64_to_felt;
use winterfell::{
    Air, AirContext, Assertion, ByteWriter, EvaluationFrame, ProofOptions, TraceInfo, TransitionConstraintDegree
};
use winter_utils::Serializable;
use winterfell::math::{FieldElement, ToElements};
use winterfell::math::fields::f128::BaseElement as Felt;


#[derive(Clone)]
pub struct TrainingUpdateInputs {
    pub initial: Vec<Felt>,    // Flattened initial state: weights then biases
    pub final_state: Vec<Felt>, // Flattened updated state
    pub steps: usize,
    pub x: Vec<Felt>,          // Input feature vector (length FE)
    pub y: Vec<Felt>,          // One‑hot label (length AC)
    pub learning_rate: Felt,   // Learning rate
    pub precision: Felt,       // Precision scaling factor
}

impl Serializable for TrainingUpdateInputs {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        for val in &self.initial {
            target.write(*val);
        }
        for val in &self.final_state {
            target.write(*val);
        }
        target.write(f64_to_felt(self.steps as f64));
    }
}

impl ToElements<Felt> for TrainingUpdateInputs {
    fn to_elements(&self) -> Vec<Felt> {
        let mut elems = self.initial.clone();
        elems.extend(self.final_state.clone());
        elems.push(f64_to_felt(self.steps as f64));
        elems
    }
}


/// AIR for the training update circuit.
pub struct TrainingUpdateAir {
    context: AirContext<Felt>,
    pub_inputs: TrainingUpdateInputs,
}

impl Air for TrainingUpdateAir {
    type BaseField = Felt;
    type PublicInputs = TrainingUpdateInputs;

    fn new(trace_info: TraceInfo, pub_inputs: TrainingUpdateInputs, options: ProofOptions) -> Self {
        let state_width = pub_inputs.initial.len();
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
        let learning_rate: E = E::from(self.pub_inputs.learning_rate);
        let precision: E = E::from(self.pub_inputs.precision);
        let fe = self.pub_inputs.x.len();
        let ac = self.pub_inputs.y.len();
        let two: E = E::from(f64_to_felt(2.0));

        for j in 0..ac {
            let mut dot: E = E::from(f64_to_felt(0.0));
            for i in 0..fe {
                let weight_index = j * fe + i;
                let x_i: E = E::from(self.pub_inputs.x[i]);
                dot = dot + frame.current()[weight_index] * x_i;
            }
            let bias_index = ac * fe + j;
            let pred = dot / precision + frame.current()[bias_index];
            let y_val: E = E::from(self.pub_inputs.y[j]);
            let error = pred - y_val;
            for i in 0..fe {
                let idx = j * fe + i;
                let x_i: E = E::from(self.pub_inputs.x[i]);
                let update_term = learning_rate * two * error * x_i / precision;
                let expected = frame.current()[idx] - update_term;
                result[idx] = frame.next()[idx] - expected;
            }
            let update_bias = learning_rate * two * error;
            let expected_bias = frame.current()[bias_index] - update_bias;
            result[bias_index] = frame.next()[bias_index] - expected_bias;
        }
        for i in (ac * fe + ac)..result.len() {
            result[i] = E::from(f64_to_felt(0.0));
        }
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let trace_len = self.trace_length();
        self.pub_inputs.final_state
            .iter()
            .enumerate()
            .map(|(i, &val)| Assertion::single(i, trace_len - 1, val))
            .collect()
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}
