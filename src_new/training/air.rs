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
    let one: E = E::ONE;
    let max_e: E = E::from(Felt::new(MAX));
    // a_cleansed = (1 - s_a)*a + s_a*(MAX - a + 1)
    let a_cleansed = (one - s_a) * a + s_a * (max_e - a + one);
    let b_cleansed = (one - s_b) * b + s_b * (max_e - b + one);
    // Use polynomial blending to compute c:
    let indicator = s_a * s_b; // Indicator is 1 if both s_a and s_b are 1, else 0.
    let c: E = indicator * (max_e + one - a_cleansed - b_cleansed)
        + (one - indicator) * (a + b);
    // Now compute the sign of c in the same way as in the helper:
    // This ideally uses a comparison gadget. For now we call compare_gadget.
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
    let one: E = E::ONE;
    let max_e: E = E::from(Felt::new(MAX));
    // Compute cleansed values in the same way.
    let a_cleansed = (one - s_a) * a + s_a * (max_e - a + one);
    let b_cleansed = (one - s_b) * b + s_b * (max_e - b + one);
    // Define the condition as: if a_sign == 0 and b_sign == 1 then condition == 1, else 0.
    // This can be computed as: (1 - s_a) * s_b.
    let condition = (one - s_a) * s_b;
    // Branch accordingly: if the condition is 1, use addition of cleansed values; otherwise, use subtraction.
    let c: E = condition * (a_cleansed + b_cleansed) + (one - condition) * (a - b);
    // Compute the result sign by comparing c with THRESHOLD.
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
    // Use E::ONE as the true multiplicative identity.
    let one: E = E::ONE;
    // Represent MAX as a field element.
    let max_e: E = E::from(Felt::new(MAX));
    // Compute cleansed values: if sign is 0, use the original; if nonzero, use (MAX - value + ONE).
    let a_cleansed = (one - s_a) * a + s_a * (max_e - a + one);
    let b_cleansed = (one - s_b) * b + s_b * (max_e - b + one);
    // Multiply the cleansed values.
    let res = a_cleansed * b_cleansed;
    // Determine the sign:
    // If the sign bits are equal or if the product is zero, then the sign is zero; else, the sign is one.
    let sign = if s_a == s_b || res == E::from(Felt::ZERO) { one - one } else { one };
    // Compute the final result:
    // If sign is zero then use res; if sign is one then use (MAX - res + ONE).
    let res_final = if sign == one - one { res } else { max_e - res + one };
    (res_final, sign)
}

/// Polynomial gadget for signed division with cleansing.
fn poly_divide<E: FieldElement<BaseField = Felt> + From<Felt> + Copy>(
    a: E,
    b: E,
    s_a: E,
    s_b: E,
) -> (E, E) {
    let one: E = E::ONE;
    let max_e: E = E::from(Felt::new(MAX));
    let a_cleansed = (one - s_a) * a + s_a * (max_e - a + one);
    let b_cleansed = (one - s_b) * b + s_b * (max_e - b + one);
    // In a field, division can be performed by multiplication with the inverse.
    // Here, we assume b_cleansed has an inverse.
    let b_inv = b_cleansed.inv(); 
    let res = a_cleansed * b_inv;
    let indicator = if s_a == s_b || res == E::from(f64_to_felt(0.0)) { one - one } else { one };
    let res_final = indicator * (max_e + one - res) + (one - indicator) * res;
    //let s_res = compare_gadget(res_final);
    let sign = if s_a == s_b || res == E::from(Felt::ZERO) { E::from(Felt::ZERO) } else { E::from(Felt::ONE) };

    (res_final, sign)
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
        // Convert public parameters to field type E.
        let learning_rate: E = E::from(self.pub_inputs.learning_rate);
        let pr: E = E::from(self.pub_inputs.precision);
        let fe = self.pub_inputs.x.len();
        let ac = self.pub_inputs.y.len();
        let one: E = E::ONE;
        // s_zero represents the sign bit 0.
        let s_zero: E = one - one;
    
        // --------------- Forward Propagation: Compute dot, pred, error ---------------
        // We'll compute the dot product for each activation using poly_multiply and poly_add.
        // Then, we divide by pr and add the bias (via poly_add) to get pred.
        // Next, we compute the basic error (pred - y) using poly_subtract,
        // and then scale the error by 2/ac using poly_multiply and poly_divide.
        // We store the computed error (with its sign) for each activation.
        let mut error_vals = vec![E::from(f64_to_felt(0.0)); ac];
        let mut s_error_vals = vec![s_zero; ac];
    
        for j in 0..ac {
            // Initialize dot = 0 with sign = 0.
            let mut dot: E = E::from(f64_to_felt(0.0));
            let mut s_dot: E = s_zero;
            // For each feature i, accumulate poly_multiply(w, x) using poly_add.
            for i in 0..fe {
                let weight_index = j * fe + i;
                let w_val: E = frame.current()[weight_index];
                let s_w: E = s_zero; // assume sign of weight is 0
                let x_i: E = E::from(self.pub_inputs.x[i]);
                let s_x: E = s_zero; // assume sign of x is 0
    
                let (prod, s_prod) = poly_multiply(w_val, x_i, s_w, s_x);
                let (new_dot, s_new_dot) = poly_add(dot, prod, s_dot, s_prod);
                dot = new_dot;
                s_dot = s_new_dot;
                            // Debug print only for activation 0, feature 0
                if j == 0 && i == 0 {
                    println!("DEBUG FOR ACTIVATION 0, FEATURE 0:");
                    println!("  w[0][0] = {:?}", w_val);
                    println!("  x[0] = {:?}", x_i);
                    println!("  prod = {:?}", prod);
                }
            }
            // Divide dot by pr using poly_divide.
            let (div_dot, s_div_dot) = poly_divide(dot, pr, s_dot, s_zero);
            // Get current bias for activation j:
            let bias_index = ac * fe + j;
            let bias_val: E = frame.current()[bias_index];
            let s_bias: E = s_zero; // assume bias sign is 0
            // Add the bias:
            let (pred, s_pred) = poly_add(div_dot, bias_val, s_div_dot, s_bias);
            // Compute basic_error = pred - y using poly_subtract.
            let y_val: E = E::from(self.pub_inputs.y[j]);
            let (basic_error, s_basic_error) = poly_subtract(pred, y_val, s_pred, s_zero);
            // Scale error by (2 / ac). First, multiply by 2:
            let (two, s_two) = (E::from(f64_to_felt(2.0)), s_zero);
            let (num, s_num) = poly_multiply(basic_error, two, s_basic_error, s_two);
            let ac_f: E = E::from(f64_to_felt(ac as f64));
            let (scaled_error, s_scaled_error) = poly_divide(num, ac_f, s_num, s_zero);
            error_vals[j] = scaled_error;
            s_error_vals[j] = s_scaled_error;
                    // Debug print for activation 0: print pred and error.
            if j == 0 {
                println!("DEBUG for activation 0:");
                println!("  dot = {:?}", dot);
                println!("  div_dot = {:?}", div_dot);
                println!("  bias = {:?}", bias_val);
                println!("  pred = {:?}", pred);
                println!("  y[0] = {:?}", y_val);
                println!("  basic_error = {:?}", basic_error);
                println!("  scaled_error = {:?}", scaled_error);
            }
        }
    
        // --------------- Backward Propagation: Enforce weight and bias updates ---------------
        // For each activation j and each feature i:
        for j in 0..ac {
            for i in 0..fe {
                let idx = j * fe + i;
                println!("  Cell ({}, {}): current = {:?}, next = {:?}, x = {:?}", 
                0, i, frame.current()[idx], frame.next()[idx], E::from(self.pub_inputs.x[i]));
                let current = frame.current()[idx];
                let s_current = s_zero; // assume sign is 0
                let x_i: E = E::from(self.pub_inputs.x[i]);
                let s_x = s_zero; // assume sign of x is 0
                let error_val = error_vals[j];
                let s_error = s_error_vals[j];
    
                // Compute: prod = poly_multiply(error, x_i)
                let (prod, s_prod) = poly_multiply(error_val, x_i, s_error, s_x);
                // Divide prod by learning_rate:
                let (temp, s_temp) = poly_divide(prod, learning_rate, s_prod, s_zero);
                // Divide temp by pr:
                let (grad, s_grad) = poly_divide(temp, pr, s_temp, s_zero);
                // Compute expected = current - grad using poly_subtract.
                let (expected, _s_expected) = poly_subtract(current, grad, s_current, s_grad);

                // Debug prints for activation 0, cell 0.
                if j == 0 && i == 0 {
                    println!("DEBUG for cell (0,0):");
                    println!("  current = {:?}", current);
                    println!("  grad = {:?}", grad);
                    println!("  expected = {:?}", expected);
                    println!("  next = {:?}", frame.next()[idx]);
                    println!("  difference = {:?}", frame.next()[idx] - expected);
                }


                // Enforce constraint: frame.next()[idx] should equal expected.
                result[idx] = frame.next()[idx] - expected;
            }
            // For the bias cell (index = ac * fe + j):
            let bias_index = ac * fe + j;
            println!("  Bias cell: current = {:?}, next = {:?}", 
            frame.current()[bias_index], frame.next()[bias_index]);
            let current_bias = frame.current()[bias_index];
            let s_current_bias = s_zero;
            // Divide error by learning_rate:
            let (temp_bias, s_temp_bias) = poly_divide(error_vals[j], learning_rate, s_error_vals[j], s_zero);
            // Compute expected_bias = current_bias - temp_bias using poly_subtract.
            let (expected_bias, _s_expected_bias) = poly_subtract(current_bias, temp_bias, s_current_bias, s_temp_bias);
            
             // Debug print for bias cell for activation 0.
            if j == 0 {
                println!("DEBUG for bias cell of activation 0:");
                println!("  current bias = {:?}", current_bias);
                println!("  expected bias = {:?}", expected_bias);
                println!("  next bias = {:?}", frame.next()[bias_index]);
                println!("  difference = {:?}", frame.next()[bias_index] - expected_bias);
            }
            
            result[bias_index] = frame.next()[bias_index] - expected_bias;
        }
        // For any extra state cells, enforce that they remain zero.
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

#[cfg(test)]
mod tests {
    use super::*; // Imports everything from this file
    use winterfell::math::{fields::f128::BaseElement as Felt, StarkField};

    // Helper: a function to compare two Felt values.
    // You might prefer to use assert_eq! if Felt implements PartialEq.
    fn assert_felt_eq(a: Felt, b: Felt) {
        assert_eq!(a.as_int(), b.as_int(), "Expected {:?}, got {:?}", b, a);
    }

    #[test]
    fn test_poly_add_zero_sign() {
        // Use simple numbers.
        let a = f64_to_felt(3.0);  // 3000000
        let b = f64_to_felt(4.0);  // 4000000
        let (res, s_res) = poly_add::<Felt>(a, b, Felt::ZERO, Felt::ZERO);
        // Expected: a + b (with sign 0)
        let expected = a + b;
        assert_felt_eq(res, expected);
        assert_eq!(s_res, Felt::ZERO, "Expected sign zero for poly_add");
    }

    #[test]
    fn test_poly_subtract_zero_sign() {
        let a = f64_to_felt(10.0); // 10e6
        let b = f64_to_felt(4.0);  // 4e6
        let (res, s_res) = poly_subtract::<Felt>(a, b, Felt::ZERO, Felt::ZERO);
        let expected = a - b;
        assert_felt_eq(res, expected);
        assert_eq!(s_res, Felt::ZERO, "Expected sign zero for poly_subtract");
    }

    #[test]
    fn test_poly_multiply_zero_sign() {
        let a = f64_to_felt(3.0);
        let b = f64_to_felt(4.0);
        let (res, s_res) = poly_multiply::<Felt>(a, b, Felt::ZERO, Felt::ZERO);
        let expected = a * b;
        assert_felt_eq(res, expected);
        assert_eq!(s_res, Felt::ZERO, "Expected sign zero for poly_multiply");
    }

    #[test]
    fn test_poly_divide_zero_sign() {
        // Let a = 12, b = 4.
        let a = f64_to_felt(12.0);
        let b = f64_to_felt(4.0);
        // In normal arithmetic, 12/4 = 3 (all numbers are scaled by 1e6).
        let (res, s_res) = poly_divide::<Felt>(a, b, Felt::ZERO, Felt::ZERO);
        let expected = a / b;
        assert_felt_eq(res, expected);
        assert_eq!(s_res, Felt::ZERO, "Expected sign zero for poly_divide");
    }
}

