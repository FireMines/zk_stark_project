// src/training/air.rs

use crate::helper::f64_to_felt;
use winterfell::{
    Air, AirContext, Assertion, ByteWriter, EvaluationFrame, ProofOptions, TraceInfo,
    TransitionConstraintDegree,
};
use winter_utils::Serializable;
use winterfell::math::{FieldElement, ToElements};
use winterfell::math::fields::f128::BaseElement as Felt;

/// Global constants used for the signed arithmetic gadgets.
/// These must match the definitions in helper.rs.
const MAX: u128 = u128::MAX; // i.e. 2^128 - 1
const THRESHOLD: u128 = 170141183460469231731687303715884105727; // example: 2^127 - 1

/// A placeholder comparison gadget.
/// In theory, this gadget outputs 1 if the input value is greater than THRESHOLD, and 0 otherwise.
/// In practice, such a gadget requires bit decomposition and many additional constraints.
/// Here we simply return 0 as a stub (indicating “false”) so that the result of the gadget is always 0.
/// (This is a placeholder and must be replaced with a true low-degree comparison gadget.)
fn compare_gadget<E: FieldElement<BaseField = Felt>>(value: E) -> E {
    // TODO: Replace with a full implementation using bit-decomposition.
    E::from(f64_to_felt(0.0))
}

/// Polynomial gadget for signed addition with cleansing.
/// This function re-implements the helper function `add` in an algebraic way.
/// Inputs:
///   a, b: field elements to add.
///   s_a, s_b: their corresponding sign bits (0 or 1, as field elements).
/// Output:
///   (c, s_c), where c is the result and s_c is the sign of the result.
/// See helper.rs for details.
fn poly_add<E: FieldElement<BaseField = Felt> + From<Felt>>(
    a: E,
    b: E,
    s_a: E,
    s_b: E,
) -> (E, E) {
    let one: E = E::from(f64_to_felt(1.0));
    let max_e: E = E::from(Felt::new(MAX));
    // a_cleansed = (1 - s_a)*a + s_a*(MAX - a + 1)
    let a_cleansed = (one - s_a) * a + s_a * (max_e - a + one);
    let b_cleansed = (one - s_b) * b + s_b * (max_e - b + one);
    // If both s_a and s_b are 1 then use the alternative formula, else use a + b.
    let indicator = s_a * s_b; // equals 1 if both are 1, else 0.
    let c = indicator * (max_e + one - a_cleansed - b_cleansed) + (one - indicator) * (a + b);
    let s_c = compare_gadget(c);
    (c, s_c)
}

/// Polynomial gadget for signed subtraction with cleansing.
fn poly_subtract<E: FieldElement<BaseField = Felt> + From<Felt>>(
    a: E,
    b: E,
    s_a: E,
    s_b: E,
) -> (E, E) {
    let one: E = E::from(f64_to_felt(1.0));
    let max_e: E = E::from(Felt::new(MAX));
    let a_cleansed = (one - s_a) * a + s_a * (max_e - a + one);
    let b_cleansed = (one - s_b) * b + s_b * (max_e - b + one);
    // If s_a != s_b and s_a == 0, then use a_cleansed + b_cleansed; else use a - b.
    let condition = (one - s_a) * (one - s_b); // equals 1 if both s_a and s_b are 0.
    let c = condition * (a_cleansed + b_cleansed) + (one - condition) * (a - b);
    let s_c = compare_gadget(c);
    (c, s_c)
}

/// Polynomial gadget for signed multiplication with cleansing.
fn poly_multiply<E: FieldElement<BaseField = Felt> + From<Felt>>(
    a: E,
    b: E,
    s_a: E,
    s_b: E,
) -> (E, E) {
    let one: E = E::from(f64_to_felt(1.0));
    let max_e: E = E::from(Felt::new(MAX));
    let a_cleansed = (one - s_a) * a + s_a * (max_e - a + one);
    let b_cleansed = (one - s_b) * b + s_b * (max_e - b + one);
    let res = a_cleansed * b_cleansed;
    // For the product sign, if s_a == s_b then result is as computed; otherwise set sign to 1 if res != 0.
    let indicator = if s_a == s_b { one - one } else { one }; // placeholder: 0 if equal, 1 if not.
    let res_final = indicator * (max_e - res + one) + (one - indicator) * res;
    let s_res = compare_gadget(res_final);
    let sign = if s_a == s_b || res == E::from(Felt::ZERO) { E::from(Felt::ZERO) } else { E::from(Felt::ONE) };

    (res_final, sign)
}

/// Polynomial gadget for signed division with cleansing.
fn poly_divide<E: FieldElement<BaseField = Felt> + From<Felt> + Copy>(
    a: E,
    b: E,
    s_a: E,
    s_b: E,
) -> (E, E) {
    let one: E = E::from(f64_to_felt(1.0));
    let max_e: E = E::from(Felt::new(MAX));
    let a_cleansed = (one - s_a) * a + s_a * (max_e - a + one);
    let b_cleansed = (one - s_b) * b + s_b * (max_e - b + one);
    // In a field, division can be performed by multiplication with the inverse.
    // Here, we assume b_cleansed has an inverse.
    let b_inv = b_cleansed.inv(); 
    let res = a_cleansed * b_inv;
    let indicator = if s_a == s_b || res == E::from(f64_to_felt(0.0)) { one - one } else { one };
    let res_final = indicator * (max_e + one - res) + (one - indicator) * res;
    let s_res = compare_gadget(res_final);
    (res_final, s_res)
}

/// Public inputs for the training update circuit.
///
/// In a production system these would be commitments to the witness,
/// but here they are given in the clear.
#[derive(Clone)]
pub struct TrainingUpdateInputs {
    /// Flattened initial state (weights then biases).
    pub initial: Vec<Felt>,
    /// Flattened final state after all training steps.
    pub final_state: Vec<Felt>,
    pub steps: usize,
    /// Input feature vector (length FE).
    pub x: Vec<Felt>,
    /// One-hot label (length AC).
    pub y: Vec<Felt>,
    /// Learning rate.
    pub learning_rate: Felt,
    /// Precision (scaling factor, pr).
    pub precision: Felt,
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
/// This AIR enforces that each transition in the execution trace follows the same update
/// as performed by the prover. Here, the update is computed using our inlined signed
/// arithmetic gadgets (poly_multiply, poly_divide, poly_subtract) to mimic backward propagation.
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

    /// Evaluate the transition constraints.
    ///
    /// For each activation j (0 ≤ j < AC) and each feature i (0 ≤ i < FE), we compute:
    ///
    /// 1. dot = sum_{i=0}^{FE-1} (current_state[j*FE+i] * x[i])
    /// 2. pred = (dot / precision) + current_state[AC*FE + j]
    /// 3. error = pred - y[j]
    ///
    /// Then for each weight cell at (j, i):
    ///
    ///   - Compute (prod, s_prod) = poly_multiply(error, x[i], s_error, s_x)
    ///   - Compute (temp, s_temp) = poly_divide(prod, learning_rate, s_prod, 0)
    ///   - Compute (grad, s_grad) = poly_divide(temp, precision, s_temp, 0)
    ///   - Compute (expected, _) = poly_subtract(current_state[j*FE+i], grad, s_current, s_grad)
    ///
    /// and enforce: frame.next()[j*FE+i] - expected == 0.
    ///
    /// Similarly for each bias at index (AC*FE + j):
    ///
    ///   - Compute (temp, s_temp) = poly_divide(error, learning_rate, s_error, 0)
    ///   - Compute (expected_bias, _) = poly_subtract(current_state[bias_index], temp, s_bias, s_temp)
    ///   - Enforce: frame.next()[bias_index] - expected_bias == 0.
    fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField> + From<Felt>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        // Convert public parameters to the generic field type E.
        let learning_rate: E = E::from(self.pub_inputs.learning_rate);
        let pr: E = E::from(self.pub_inputs.precision);
        let fe = self.pub_inputs.x.len();
        let ac = self.pub_inputs.y.len();
        let one: E = E::from(f64_to_felt(1.0));
        // s_zero represents the sign bit 0.
        let s_zero: E = one - one;
    
        // ---- Compute Forward Propagation: dot product, prediction and error ----
        // For each activation j, we compute:
        // dot = sum_{i=0}^{fe-1} poly_add(…, poly_multiply(w[j][i], x[i], 0, 0) ... );
        // Then, after dividing by pr and adding bias, we get pred.
        // Then, error is computed as basic_error * (2 / ac).
        // We'll compute these per activation.
        let mut error_per_act = vec![E::from(f64_to_felt(0.0)); ac];
        for j in 0..ac {
            // Compute dot = \sum_{i=0}^{fe-1} poly_add over poly_multiply(w, x)
            let mut dot: E = E::from(f64_to_felt(0.0));
            let mut s_dot: E = s_zero;
            for i in 0..fe {
                let weight_index = j * fe + i;
                let w_i: E = frame.current()[weight_index];
                // Convert public x[i] to E.
                let x_i: E = E::from(self.pub_inputs.x[i]);
                // Assume sign bits for w[i] and x[i] are zero.
                let (prod, s_prod) = poly_multiply(w_i, x_i, s_zero, s_zero);
                // Now add prod to current dot:
                let (new_dot, s_new_dot) = poly_add(dot, prod, s_dot, s_prod);
                dot = new_dot;
                s_dot = s_new_dot;
            }
            // Divide dot by pr.
            let (div_dot, s_div_dot) = poly_divide(dot, pr, s_dot, s_zero);
            // Get current bias for activation j (which is at index ac*fe + j in the state).
            let bias_index = ac * fe + j;
            let bias_val: E = frame.current()[bias_index];
            // Add bias to div_dot.
            let (pred, s_pred) = poly_add(div_dot, bias_val, s_div_dot, s_zero);
            // Compute basic_error = pred - y.
            let y_val: E = E::from(self.pub_inputs.y[j]);
            let (basic_error, s_basic_error) = poly_subtract(pred, y_val, s_pred, s_zero);
            // Scale error by 2/ac.
            let ac_f: E = E::from(f64_to_felt(ac as f64));
            let (two, s_two) = (E::from(f64_to_felt(2.0)), s_zero);
            let (num, s_num) = poly_multiply(basic_error, two, s_basic_error, s_two);
            let (scaled_error, s_scaled_error) = poly_divide(num, ac_f, s_num, s_zero);
            // Save error for activation j.
            error_per_act[j] = scaled_error;
        }
    
        // ---- Enforce Backward Propagation Update for Weights and Biases ----
        // For each activation j and feature i, enforce the update:
        // expected = poly_subtract(current_weight, grad, 0, s_grad)
        // where grad is obtained by:
        // grad = poly_divide( poly_divide( poly_multiply(error, x[i], 0, 0), learning_rate, 0, 0), pr, 0, 0)
        for j in 0..ac {
            for i in 0..fe {
                let idx = j * fe + i;
                let current = frame.current()[idx];
                // Use public x[i] and error for activation j.
                let x_i: E = E::from(self.pub_inputs.x[i]);
                let s_x: E = s_zero; // assume
                let s_error: E = s_zero; // assume
                let error_val = error_per_act[j];
                
                // Step 1: Multiply error and x_i using poly_multiply.
                let (prod, s_prod) = poly_multiply(error_val, x_i, s_error, s_x);
                // Step 2: Divide by learning_rate.
                let (temp, s_temp) = poly_divide(prod, learning_rate, s_prod, s_zero);
                // Step 3: Divide by pr.
                let (grad, s_grad) = poly_divide(temp, pr, s_temp, s_zero);
                // Step 4: Subtract grad from current weight.
                let (expected, _s_expected) = poly_subtract(current, grad, s_zero, s_grad);
                // The constraint: frame.next()[idx] should equal expected.
                result[idx] = frame.next()[idx] - expected;
            }
            // For each bias cell: (index = ac*fe + j)
            let bias_index = ac * fe + j;
            let current_bias = frame.current()[bias_index];
            // Compute bias update: temp_bias = poly_divide(error, learning_rate, 0, 0)
            let (temp_bias, s_temp_bias) = poly_divide(error_per_act[j], learning_rate, s_zero, s_zero);
            let (expected_bias, _s_expected_bias) = poly_subtract(current_bias, temp_bias, s_zero, s_temp_bias);
            result[bias_index] = frame.next()[bias_index] - expected_bias;
        }
        // For any extra state cells, enforce zero.
        for i in (ac * fe + ac)..result.len() {
            result[i] = E::from(f64_to_felt(0.0));
        }
    }
    
    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let trace_len = self.trace_length();
        self.pub_inputs
            .final_state
            .iter()
            .enumerate()
            .map(|(i, &val)| Assertion::single(i, trace_len - 1, val))
            .collect()
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}


