// src/helper.rs
// helper.rs (prover side)
pub use crate::signed::{add, sub as subtract, mul as multiply, div as divide};

use std::error::Error;
use csv::ReaderBuilder;
use rand::seq::index::sample;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use winterfell::math::{FieldElement, StarkField};
use winterfell::math::fields::f128::BaseElement as Felt;
use winterfell::EvaluationFrame;

// Global constants (matching ZoKrates)
// We hard-code the values to avoid calling non-const operators.
pub const MAX: u128 = u128::MAX; // 340_282_366_920_938_463_463_374_607_431_768_211_455u128
const THRESHOLD: Felt = Felt::new(170141183460469231731687303715884105727); // about 2^127 - 1
pub const AC: usize = 6; // number of activations (layers)
const AC_F: Felt = Felt::new(6);
pub const FE: usize = 9; // number of features per activation
pub const C: usize = 8;  // number of clients
const BS: usize = 2;

/// Convert an f64 value to our field element using a scaling factor of 1e6.
pub fn f64_to_felt(x: f64) -> Felt {
    Felt::new((x * 1e6).round() as u128)
}

/// Converts a u64 value to a field element.
pub fn u64_to_field(x: u64) -> Felt {
    Felt::new(x as u128)
}

pub fn u32_to_field(x: u32) -> Felt {
    Felt::new(x as u128)
}


/// Signed addition with cleansing:
/// - For each operand, if its sign is 0 use the value as is,
///   otherwise use (MAX - value + 1).
/// - If both operands have sign 1, compute c = (MAX + 1 - a_cleansed - b_cleansed),
///   else c = a + b.
/// - Set the result’s sign to 1 if c > THRESHOLD, otherwise 0.
// helper.rs -------------------------------------------------------------

pub const fn encode_signed(x: i128) -> (Felt, Felt) {
    if x >= 0 {
        (Felt::new(x as u128), Felt::ZERO)          //  +x
    } else {
        let abs = (-x) as u128;
        (Felt::new(MAX.wrapping_sub(abs).wrapping_add(1)), Felt::ONE)   //  -x
    }
}

/// scale f64 → (value, sign)
pub fn f64_to_signed_felt(x: f64, scale: f64) -> (Felt, Felt) {
    let scaled = (x * scale).round() as i128;
    encode_signed(scaled)
}


/// Read a CSV row of length 46 or 10, extract features+label
pub fn read_dataset(
    file_path: &str,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    let mut feats = Vec::new();
    let mut labs  = Vec::new();
    for r in rdr.records() {
        let rec = r?;
        if rec.is_empty() { continue; }
        let row: Vec<f64> = rec.iter()
            .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
            .collect();
        match row.len() {
            46 => {
                feats.push(row[18..27].to_vec());
                labs.push(row[45]);
            }
            10 => {
                feats.push(row[..9].to_vec());
                labs.push(row[9]);
            }
            n => return Err(format!("Unexpected CSV width {}", n).into()),
        }
    }
    Ok((feats, labs))
}

/// A single edge device: holds all data and samples `p` rows.
pub struct EdgeDevice {
    pub features: Vec<Vec<f64>>,
    pub labels:   Vec<f64>,
}
impl EdgeDevice {
    pub fn new(f: Vec<Vec<f64>>, l: Vec<f64>) -> Self {
        EdgeDevice { features: f, labels: l }
    }

    /// Randomly sample `p` distinct rows (no replacement)
    pub fn next_batch(&self, p: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = self.labels.len();
        let p = p.min(n);
        let mut rng = thread_rng();
        let idxs = sample(&mut rng, n, p).into_vec();
        let mut bx = Vec::with_capacity(p);
        let mut by = Vec::with_capacity(p);
        for i in idxs {
            bx.push(self.features[i].clone());
            by.push(self.labels[i]);
        }
        (bx, by)
    }
}
/// Splits the dataset among a given number of clients using round‑robin.
pub fn split_dataset(
    features: Vec<Vec<f64>>,
    labels: Vec<f64>,
    num_clients: usize,
) -> Vec<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut clients = vec![(Vec::new(), Vec::new()); num_clients];
    for (i, (feat, lab)) in features.into_iter().zip(labels.into_iter()).enumerate() {
        let idx = i % num_clients;
        clients[idx].0.push(feat);
        clients[idx].1.push(lab);
    }
    clients
}

/// Generates an initial model using a normal distribution (replicates Veriblock‑FL).
pub fn generate_initial_model(
    fe: usize,
    ac: usize,
    sigma: f64                // standard‑dev in **original** real units
) -> (Vec<Vec<Felt>>, Vec<Vec<Felt>>, Vec<Felt>, Vec<Felt>) {

    let normal   = Normal::new(0.0, sigma).unwrap();
    let mut rng  = thread_rng();

    let mut w      = vec![vec![Felt::ZERO; fe]; ac];
    let mut w_sign = vec![vec![Felt::ZERO; fe]; ac];
    let mut b      = vec![Felt::ZERO; ac];
    let mut b_sign = vec![Felt::ZERO; ac];

    for j in 0..ac {
        for i in 0..fe {
            let (v, s) = f64_to_signed_felt(normal.sample(&mut rng), 1e6);
            w[j][i] = v;  w_sign[j][i] = s;
        }
        let (v, s) = f64_to_signed_felt(normal.sample(&mut rng), 1e6);
        b[j] = v;  b_sign[j] = s;
    }
    (w, w_sign, b, b_sign)
}



/// Extension trait to get current step from evaluation frame
pub trait EvaluationFrameExt<E: FieldElement> {
    fn current_step(&self) -> usize;
}

impl<E: FieldElement> EvaluationFrameExt<E> for EvaluationFrame<E> {
    fn current_step(&self) -> usize {
        // This is a simplified implementation - you might need to adjust 
        // based on how winterfell tracks steps internally
        // For now, we'll assume the step is tracked externally or inferred
        0 // Placeholder - you might need to modify this based on winterfell's API
    }
}

/// Process a batch of samples for training update
pub fn process_batch_update(
    w: &mut Vec<Vec<Felt>>,
    b: &mut Vec<Felt>,
    w_sign: &mut Vec<Vec<Felt>>,
    b_sign: &mut Vec<Felt>,
    x_batch: &[Vec<Felt>],
    x_batch_sign: &[Vec<Felt>],
    y_batch: &[Vec<Felt>],
    learning_rate: Felt,
    precision: Felt,
) -> (Vec<Vec<Felt>>, Vec<Felt>, Vec<Vec<Felt>>, Vec<Felt>) {
    
    for sample_idx in 0..x_batch.len() {
        // Forward pass for this sample
        let (output, output_sign) = forward_propagation_layer(
            w, b, &x_batch[sample_idx],
            w_sign, b_sign, &x_batch_sign[sample_idx],
            precision,
        );
        
        // Compute error for this sample
        let (error, error_sign) = mse_prime(
            &y_batch[sample_idx], &output, &output_sign, precision
        );
        
        // Backward pass for this sample
        let (new_w, new_b, new_w_sign, new_b_sign) = backward_propagation_layer(
            w, b, &x_batch[sample_idx], &error,
            learning_rate, precision,
            w_sign, b_sign,
            &x_batch_sign[sample_idx], &error_sign,
        );
        
        // Update weights and biases in place
        *w = new_w;
        *b = new_b;
        *w_sign = new_w_sign;
        *b_sign = new_b_sign;
    }
    
    (w.clone(), b.clone(), w_sign.clone(), b_sign.clone())
}

/// Convert a batch of labels to one-hot encoding
pub fn batch_to_one_hot(labels: &[f64], ac: usize, precision: f64) -> Vec<Vec<Felt>> {
    labels.iter().map(|&label| {
        let (onehot, _) = label_to_one_hot(label, ac, precision);
        onehot
    }).collect()
}

/// Converts a scalar label to a one‑hot vector of length AC.
pub fn label_to_one_hot(label: f64, ac: usize, precision: f64)
    -> (Vec<Felt>, Vec<Felt>)
{
    let mut v      = vec![Felt::ZERO; ac];
    let mut v_sign = vec![Felt::ZERO; ac];

    let idx = if label < 1.0 { 0 } else { (label as usize).saturating_sub(1) };
    if idx < ac {
        let (val, sig) = f64_to_signed_felt(precision, 1.0);
        v[idx] = val;  v_sign[idx] = sig;
    }
    (v, v_sign)
}

/// Flattens a weight matrix and bias vector into a single vector.
pub fn flatten_state_matrix(w: &Vec<Vec<Felt>>, b: &Vec<Felt>) -> Vec<Felt> {
    let mut flat = Vec::new();
    for row in w {
        flat.extend_from_slice(row);
    }
    flat.extend_from_slice(b);
    flat
}

// when you flatten state into a row:
pub fn flatten_with_sign(w: &[Vec<Felt>], w_sign: &[Vec<Felt>],
        b: &[Felt], b_sign: &[Felt]) -> Vec<Felt> {
    let mut v = Vec::with_capacity(2*(w.len()*w[0].len() + b.len()));
    for (row, srow) in w.iter().zip(w_sign) {
        for (&v_i, &s_i) in row.iter().zip(srow) {
            v.push(v_i);   v.push(s_i);
        }
    }
    for (&v_i, &s_i) in b.iter().zip(b_sign) {
        v.push(v_i);  v.push(s_i);
    }
    v
}


/// Unflattens a state vector into a weight matrix and bias vector.
/// Assumes that the first (AC * FE) elements are the weights (row‑major)
/// and the remaining AC elements are the biases.
pub fn unflatten_state_matrix(
    state: &Vec<Felt>,
    ac: usize,
    fe: usize,
) -> (Vec<Vec<Felt>>, Vec<Felt>) {
    let weights: Vec<Vec<Felt>> = (0..ac)
        .map(|j| state[(j * fe)..(j * fe + fe)].to_vec())
        .collect();
    let biases: Vec<Felt> = state[(ac * fe)..].to_vec();
    (weights, biases)
}

/// From a `[v0,s0,v1,s1,…]` row build (w, b) and their sign matrices.
pub fn split_state_with_sign(
    row: &[Felt], ac: usize, fe: usize,
) -> (Vec<Vec<Felt>>, Vec<Felt>, Vec<Vec<Felt>>, Vec<Felt>) {
    let expected = 2 * ac * (fe + 1);
    assert_eq!(
        row.len(), expected,
        "split_state_with_sign: expected row.len() = {}, got {}",
        expected, row.len()
    );
    let mut w      = vec![vec![Felt::ZERO; fe]; ac];
    let mut w_sign = vec![vec![Felt::ZERO; fe]; ac];
    let mut b      = vec![Felt::ZERO; ac];
    let mut b_sign = vec![Felt::ZERO; ac];

    // weights ---------------------------------------------------------------
    for j in 0..ac {
        for i in 0..fe {
            let idx = 2 * (j*fe + i);
            w[j][i]      = row[idx];
            w_sign[j][i] = row[idx + 1];
        }
    }
    // biases ----------------------------------------------------------------
    for j in 0..ac {
        let idx = 2 * (ac*fe + j);
        b[j]      = row[idx];
        b_sign[j] = row[idx + 1];
    }
    (w, b, w_sign, b_sign)
}


/// Extract only the *values* (skip every second “sign” column) from a
/// flattened `[val,sign,…]` row layout.
///
/// Returns `(weights, biases)` where
/// * `weights` has shape `[ac][fe]`
/// * `biases`  has length `ac`
pub fn unflatten_values_only(
    state: &[Felt],
    ac: usize,
    fe: usize,
) -> (Vec<Vec<Felt>>, Vec<Felt>) {
    let mut w = vec![vec![Felt::ZERO; fe]; ac];
    let mut idx = 0;
    for j in 0..ac {
        for i in 0..fe {
            w[j][i] = state[idx];     // take the value
            idx += 2;                 // skip its sign
        }
    }
    let mut b = vec![Felt::ZERO; ac];
    for j in 0..ac {
        b[j] = state[idx];
        idx += 2;                     // skip sign
    }
    (w, b)
}


/// Transposes a 2D vector.
pub fn transpose(matrix: Vec<Vec<Felt>>) -> Vec<Vec<Felt>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![f64_to_felt(0.0); rows]; cols];
    for i in 0..rows {
        assert_eq!(matrix[i].len(), cols, "All rows must have equal length");
        for j in 0..cols {
            transposed[j][i] = matrix[i][j];
        }
    }
    transposed
}

pub fn mimc_cipher(input: Felt, round_constant: Felt, z: Felt) -> Felt {
    let mut inp = input;
    for _ in 0..64 {
        let a = inp + round_constant + z;
        inp = <Felt as FieldElement>::exp(a, 7);
    }
    inp + z
}

pub fn mimc_hash_matrix(w: &[Vec<Felt>], b: &[Felt], round_constants: &[Felt]) -> Felt {
    let mut z = f64_to_felt(0.0);
    for i in 0..w.len() {
        for j in 0..w[i].len() {
            let rc = round_constants[j % round_constants.len()];
            z = mimc_cipher(w[i][j], rc, z);
        }
        let rc = round_constants[i % round_constants.len()];
        z = mimc_cipher(b[i], rc, z);
    }
    z
}

/// Computes the derivative of the mean squared error (MSE′) for a layer.
///
/// # Parameters
/// - `y_true`: True output vector (length ac)
/// - `y_pred`: Predicted output vector (length ac)
/// - `y_pred_sign`: Sign vector for the predicted outputs (length ac)
/// - `pr`: A precision or scaling parameter (unused in this function body)
///
/// # Returns
/// A tuple `(result, result_sign)` representing the gradient with respect to y_pred.
pub fn mse_prime(
    y_true: &[Felt],
    y_pred: &[Felt],
    y_pred_sign: &[Felt],
    pr: Felt,
) -> (Vec<Felt>, Vec<Felt>) {
    let ac = y_true.len();
    // Represent the number of activations as a field element.
    let ac_f = f64_to_felt(ac as f64);

    let mut result = vec![Felt::ZERO; ac];
    let mut result_sign = vec![Felt::ZERO; ac];

    for i in 0..ac {
        // Compute the difference: (y_pred - y_true)
        let (temp, temp_sign) = subtract(y_pred[i], y_true[i], y_pred_sign[i], Felt::ZERO);
        // Multiply the difference by 2: 2 * (y_pred - y_true)
        let (temp2, temp2_sign) = multiply(temp, f64_to_felt(2.0), temp_sign, Felt::ZERO);
        // Divide by the number of activations: [2 * (y_pred - y_true)] / ac_f
        let (res, res_sign) = divide(temp2, ac_f, temp2_sign, Felt::ZERO);
        result[i] = res;
        result_sign[i] = res_sign;
    }

    (result, result_sign)
}


/// Computes a forward propagation layer.
/// 
/// - `w`: Weight matrix of shape [ac][fe]
/// - `b`: Bias vector of length ac
/// - `x`: Input vector of length fe
/// - `w_sign`, `b_sign`, `x_sign`: Corresponding sign values for w, b, and x
/// - `pr`: A scaling or precision parameter
/// 
/// Returns a tuple (result, result_sign) of length ac.
pub fn forward_propagation_layer(
    w: &[Vec<Felt>],
    b: &[Felt],
    x: &[Felt],
    w_sign: &[Vec<Felt>],
    b_sign: &[Felt],
    x_sign: &[Felt],
    pr: Felt,
) -> (Vec<Felt>, Vec<Felt>) {
    // Assume that the number of activations (ac) is given by the length of b.
    // And the number of features (fe) is the length of x.
    let ac = b.len();
    let fe = x.len();

    // This vector will hold the intermediate weighted sum results.
    let mut wx = vec![Felt::ZERO; ac];
    let mut wx_sign = vec![Felt::ZERO; ac];

    // For each activation unit:
    for j in 0..ac {
        let mut temp = Felt::ZERO;
        let mut temp_sign = Felt::ZERO;
        // Sum over the features.
        for i in 0..fe {
            let (t_i, t_i_s) = multiply(w[j][i], x[i], w_sign[j][i], x_sign[i]);
            let (new_temp, new_temp_sign) = add(temp, t_i, temp_sign, t_i_s);
            temp = new_temp;
            temp_sign = new_temp_sign;
        }
        // Divide the accumulated sum by the precision/scaling parameter.
        let (div_temp, div_temp_sign) = divide(temp, pr, temp_sign, Felt::ZERO);
        wx[j] = div_temp;
        wx_sign[j] = div_temp_sign;
    }

    // Finally, add the bias to each activation.
    let mut result = vec![Felt::ZERO; ac];
    let mut result_sign = vec![Felt::ZERO; ac];
    for j in 0..ac {
        let (res, res_sign) = add(wx[j], b[j], wx_sign[j], b_sign[j]);
        result[j] = res;
        result_sign[j] = res_sign;
    }

    (result, result_sign)
}

/// Performs the backward propagation update for a layer.
///
/// # Parameters
/// - `w`: Mutable weight matrix of shape [ac][fe]
/// - `b`: Mutable bias vector of length ac
/// - `x`: Input vector of length fe (used for gradient computation)
/// - `output_error`: Vector of errors for each activation (length ac)
/// - `learning_rate`: Learning rate (scalar)
/// - `pr`: Precision or scaling parameter (scalar)
/// - `w_sign`: Mutable matrix with sign values for weights (shape [ac][fe])
/// - `b_sign`: Mutable vector with sign values for biases (length ac)
/// - `x_sign`: Vector with sign values for inputs (length fe)
/// - `output_error_sign`: Vector with sign values for output error (length ac)
///
/// # Returns
/// A tuple containing the updated `(w, b, w_sign, b_sign)`.
pub fn backward_propagation_layer(
    w: &mut Vec<Vec<Felt>>,
    b: &mut Vec<Felt>,
    x: &[Felt],
    output_error: &[Felt],
    learning_rate: Felt,
    pr: Felt,
    w_sign: &mut Vec<Vec<Felt>>,
    b_sign: &mut Vec<Felt>,
    x_sign: &[Felt],
    output_error_sign: &[Felt],
) -> (Vec<Vec<Felt>>, Vec<Felt>, Vec<Vec<Felt>>, Vec<Felt>) {
    // Determine dimensions: number of activations and number of features.
    let ac = b.len();
    let fe = x.len();

    // --- Update the Bias Vector ---
    for i in 0..ac {
        let (temp, temp_sign) = divide(output_error[i], learning_rate, output_error_sign[i], Felt::ZERO);
        let (new_b, new_b_sign) = subtract(b[i], temp, b_sign[i], temp_sign);
        b[i] = new_b;
        b_sign[i] = new_b_sign;
    }

    // --- Update the Weight Matrix ---
    for j in 0..fe {
        for i in 0..ac {
            // Multiply the output error for activation i by x[j].
            let (prod, prod_sign) = multiply(output_error[i], x[j], output_error_sign[i], x_sign[j]);
            // Divide the product by the learning rate.
            let (temp, temp_sign) = divide(prod, learning_rate, prod_sign, Felt::ZERO);
            // Divide that result by pr (precision/scaling).
            let (grad, grad_sign) = divide(temp, pr, temp_sign, Felt::ZERO);

            // For debugging: Print detailed info for a specific cell,
            // here choose activation 1, feature 0 (i==1, j==0) as an example.
           /* if i == 2 && j == 0 {
                let current_weight = w[i][j];
                let (expected, _expected_sign) = subtract(current_weight, grad, w_sign[i][j], grad_sign);
                println!("--- Detailed Debug for activation {}, feature {} ---", i, j);
                println!("  current_weight = {:?}", current_weight.as_int());
                println!("  x = {:?}", x[j].as_int());
                println!("  prod = {:?}", prod.as_int());
                println!("  temp = {:?}", temp.as_int());
                println!("  grad = {:?}", grad.as_int());
                println!("  expected new weight = {:?}", expected.as_int());
            }*/
            
            // Subtract the gradient from the current weight.
            let (new_w, new_w_sign) = subtract(w[i][j], grad, w_sign[i][j], grad_sign);
            w[i][j] = new_w;
            w_sign[i][j] = new_w_sign;
                    }
    }

    (w.clone(), b.clone(), w_sign.clone(), b_sign.clone())
}


pub fn get_round_constants() -> Vec<Felt> {
    (1..=64).map(|i| f64_to_felt(i as f64)).collect()
}


/// Updates global weights and biases based on client-provided local updates.
/// 
/// # Parameters
/// - `global_w` and `global_w_sign`: Global weight matrix and its sign (dimensions AC x FE).
/// - `global_b` and `global_b_sign`: Global bias vector and its sign (length AC).
/// - `local_w` and `local_w_sign`: Local weight matrices and their signs from each client (dimensions c x AC x FE).
/// - `local_b` and `local_b_sign`: Local bias vectors and their signs from each client (dimensions c x AC).
/// 
/// # Returns
/// A tuple containing the new global weights, biases and their sign matrices:
/// `(new_global_w, new_global_b, new_global_w_sign, new_global_b_sign)`.
pub fn update_global(
    global_w: &Vec<Vec<Felt>>, 
    global_w_sign: &Vec<Vec<Felt>>,
    global_b: &Vec<Felt>, 
    global_b_sign: &Vec<Felt>,
    local_w: &Vec<Vec<Vec<Felt>>>, 
    local_w_sign: &Vec<Vec<Vec<Felt>>>,
    local_b: &Vec<Vec<Felt>>, 
    local_b_sign: &Vec<Vec<Felt>>
) -> (Vec<Vec<Felt>>, Vec<Felt>, Vec<Vec<Felt>>, Vec<Felt>) {
    // Initialize new global parameters with zeros.
    let mut new_global_w = vec![vec![Felt::ZERO; FE]; AC];
    let mut new_global_b = vec![Felt::ZERO; AC];
    let mut new_global_w_sign = vec![vec![Felt::ZERO; FE]; AC];
    let mut new_global_b_sign = vec![Felt::ZERO; AC];

    // The number of clients (c) is the length of the first dimension of local_w.
    let c = local_w.len();
    let k_field = u32_to_field(c as u32); // Convert c to a field element.

    // Iterate over each client.
    for client in 0..c {
        // Update global weights.
        for i in 0..AC {
            for j in 0..FE {
                // Compute the difference between the client's local weight and the global weight.
                let (temp, temp_sign) = subtract(
                    local_w[client][i][j],
                    global_w[i][j],
                    local_w_sign[client][i][j],
                    global_w_sign[i][j],
                );
                // Divide the difference by the number of clients.
                let (temp, temp_sign) = divide(temp, k_field, temp_sign, Felt::ZERO);
                // Add the divided difference to the current global weight.
                let (updated, updated_sign) = add(
                    global_w[i][j],
                    temp,
                    global_w_sign[i][j],
                    temp_sign,
                );
                new_global_w[i][j] = updated;
                new_global_w_sign[i][j] = updated_sign;
            }
        }

        // Update global biases.
        for i in 0..AC {
            let (temp, temp_sign) = subtract(
                local_b[client][i],
                global_b[i],
                local_b_sign[client][i],
                global_b_sign[i],
            );
            let (temp, temp_sign) = divide(temp, k_field, temp_sign, Felt::ZERO);
            let (updated, updated_sign) = add(
                global_b[i],
                temp,
                global_b_sign[i],
                temp_sign,
            );
            new_global_b[i] = updated;
            new_global_b_sign[i] = updated_sign;
        }
    }

    (new_global_w, new_global_b, new_global_w_sign, new_global_b_sign)
}



/// Checks if all the elements of `sc_lhashes` match those of `local_params_hash`.
/// 
/// Note: This implementation iterates over every pair of indices from 0 to `c` and sets
/// the flag to zero if any pair does not match. (This exactly mirrors the provided pseudocode.)
pub fn local_hash_contains(sc_lhashes: &[Felt], local_params_hash: &[Felt]) -> Felt {
    let c = sc_lhashes.len();
    let mut is_current_hash_found = Felt::ONE; // Assume all hashes match initially.
    for i in 0..c {
        for j in 0..c {
            // If any pair doesn't match, set flag to 0.
            if sc_lhashes[i] != local_params_hash[j] {
                is_current_hash_found = Felt::ZERO;
            }
        }
    }
    is_current_hash_found
}

/// Implements a MiMC cipher round function.
/// 
/// - `input`: the current field element input.
/// - `round_constants`: an array of 64 round constants.
/// - `z`: the current chaining variable.
/// 
/// For each round, it computes:
///   a = input + round_constants[i] + z,
///   then sets input = a^7.
/// Finally, it returns input + z.
pub fn mimc_cipher_aggregator(mut input: Felt, round_constants: &[Felt; 64], z: Felt) -> Felt {
    let mut a = Felt::ZERO;
    for i in 0..64 {
        a = input + round_constants[i] + z;
        // Compute a to the power of 7. Adjust the method as needed if your field type uses a different API.
        input = <Felt as FieldElement>::exp(a, 7);
    }
    input + z
}

/// Computes a MiMC hash from a weight matrix `w` and bias vector `b` using the provided round constants.
/// 
/// - `w`: a matrix of dimensions [AC][FE] (global weights).
/// - `b`: a vector of length AC (global biases).
/// - `round_constants`: an array of 64 field elements.
/// 
/// The hash is computed by iterating over each activation and each feature (processing weights)
/// and then processing each bias with the MiMC cipher.
pub fn mimc_hash_aggregator(w: &Vec<Vec<Felt>>, b: &Vec<Felt>, round_constants: &[Felt; 64]) -> Felt {
    let mut z = Felt::ZERO;
    // It is assumed that the number of activations is AC and features per activation is FE.
    // (AC and FE should be defined as constants elsewhere in your project.)
    for i in 0..w.len() {
        for j in 0..w[i].len() {
            let rc = round_constants[j % round_constants.len()];
            z = mimc_cipher(w[i][j], rc, z);
        }
        let rc = round_constants[i % round_constants.len()];
        z = mimc_cipher(b[i], rc, z);
    }
    z
}

/// Convenience: convert a bool‑like Felt (0/1) into usize
#[inline] pub fn felt_to_bool(f: Felt) -> bool { f == Felt::ONE }

/// Enforce “bit” in the AIR:  s · (s − 1) = 0
#[inline] pub fn bit_constraint<E: FieldElement>(s: E) -> E { s * (s - E::ONE) }



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
    fn test_add_zero_sign() {
        // Use simple numbers.
        let a = f64_to_felt(3.0);  // 3000000
        let b = f64_to_felt(4.0);  // 4000000
        let (res, s_res) = add(a, b, Felt::ZERO, Felt::ZERO);
        // Expected: a + b (with sign 0)
        let expected = a + b;
        assert_felt_eq(res, expected);
        assert_eq!(s_res, Felt::ZERO, "Expected sign zero for add");
    }

    #[test]
    fn test_subtract_zero_sign() {
        let a = f64_to_felt(10.0); // 10e6
        let b = f64_to_felt(4.0);  // 4e6
        let (res, s_res) = subtract(a, b, Felt::ZERO, Felt::ZERO);
        let expected = a - b;
        assert_felt_eq(res, expected);
        assert_eq!(s_res, Felt::ZERO, "Expected sign zero for subtract");
    }

    #[test]
    fn test_multiply_zero_sign() {
        let a = f64_to_felt(3.0);
        let b = f64_to_felt(4.0);
        let (res, s_res) = multiply(a, b, Felt::ZERO, Felt::ZERO);
        let expected = a * b;
        assert_felt_eq(res, expected);
        assert_eq!(s_res, Felt::ZERO, "Expected sign zero for multiply");
    }

    #[test]
    fn test_divide_zero_sign() {
        // Let a = 12, b = 4.
        let a = f64_to_felt(12.0);
        let b = f64_to_felt(4.0);
        // In normal arithmetic, 12/4 = 3 (all numbers are scaled by 1e6).
        let (res, s_res) = divide(a, b, Felt::ZERO, Felt::ZERO);
        let expected = a / b;
        assert_felt_eq(res, expected);
        assert_eq!(s_res, Felt::ZERO, "Expected sign zero for divide");
    }


    // A helper function to create a simple matrix and bias vector.
    fn example_state() -> (Vec<Vec<Felt>>, Vec<Felt>) {
        // For example, create a 2x3 weight matrix and a bias vector of length 2.
        // (Make sure to use your f64_to_felt conversion or direct `Felt::new`.)
        let w = vec![
            vec![Felt::new(1), Felt::new(2), Felt::new(3)],
            vec![Felt::new(4), Felt::new(5), Felt::new(6)],
        ];
        let b = vec![Felt::new(7), Felt::new(8)];
        (w, b)
    }

    #[test]
    fn test_flatten_unflatten() {
        let (w, b) = example_state();
        let flat = flatten_state_matrix(&w, &b);
        let (w2, b2) = unflatten_state_matrix(&flat, 2, 3); // 2 activations, 3 features
        assert_eq!(w, w2, "Unflattened weight matrix does not match the original");
        assert_eq!(b, b2, "Unflattened bias vector does not match the original");
    }

    #[test]
    fn test_transpose() {
        // Create a simple 2x3 matrix.
        let matrix = vec![
            vec![Felt::new(1), Felt::new(2), Felt::new(3)],
            vec![Felt::new(4), Felt::new(5), Felt::new(6)],
        ];
        let expected_transpose = vec![
            vec![Felt::new(1), Felt::new(4)],
            vec![Felt::new(2), Felt::new(5)],
            vec![Felt::new(3), Felt::new(6)],
        ];
        let t = transpose(matrix);
        assert_eq!(t, expected_transpose, "Transpose function did not produce the expected result");
    }

    // Test utilities
    fn felt_to_f64(f: Felt) -> f64 {
        f.as_int() as f64 / 1e6
    }

    fn compare_floats(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    // Reference implementation for comparison
    mod reference {
        pub fn forward_pass(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
            let mut result = Vec::new();
            for i in 0..w.len() {
                let mut sum = b[i];
                for j in 0..w[i].len() {
                    sum += w[i][j] * x[j];
                }
                result.push(sum);
            }
            result
        }

        pub fn mse_prime(y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
            let n = y_true.len() as f64;
            y_pred.iter()
                .zip(y_true.iter())
                .map(|(pred, true_val)| 2.0 * (pred - true_val) / n)
                .collect()
        }

        pub fn backward_pass(w: &mut Vec<Vec<f64>>, b: &mut Vec<f64>, x: &[f64], error: &[f64], lr: f64) {
            // Update biases
            for i in 0..b.len() {
                b[i] -= lr * error[i];
            }
            
            // Update weights
            for i in 0..w.len() {
                for j in 0..w[i].len() {
                    w[i][j] -= lr * error[i] * x[j];
                }
            }
        }
    }

    #[test]
    fn test_field_conversion() {
        let values = [0.0, 1.0, -1.0, 0.5, -0.5, 0.000001, -0.000001];
        for &val in &values {
            let felt = f64_to_felt(val);
            let back = felt_to_f64(felt);
            assert!(compare_floats(val, back, 1e-6), 
                   "Conversion failed for {}: {} -> {} -> {}", val, val, felt.as_int(), back);
        }
    }

    #[test]
    fn test_signed_arithmetic() {
        // Test addition
        let a = f64_to_felt(3.5);
        let b = f64_to_felt(2.1);
        let (result, sign) = add(a, b, Felt::ZERO, Felt::ZERO);
        assert!(compare_floats(felt_to_f64(result), 5.6, 1e-6));
        assert_eq!(sign, Felt::ZERO);

        // Test subtraction
        let (result, sign) = subtract(a, b, Felt::ZERO, Felt::ZERO);
        assert!(compare_floats(felt_to_f64(result), 1.4, 1e-6));
        assert_eq!(sign, Felt::ZERO);

        // Test multiplication
        let (result, sign) = multiply(a, b, Felt::ZERO, Felt::ZERO);
        assert!(compare_floats(felt_to_f64(result), 7.35, 1e-6));
        assert_eq!(sign, Felt::ZERO);

        // Test division
        let (result, sign) = divide(a, b, Felt::ZERO, Felt::ZERO);
        assert!(compare_floats(felt_to_f64(result), 1.666667, 1e-5));
        assert_eq!(sign, Felt::ZERO);
    }

    #[test]
    fn test_forward_propagation_correctness() {
        // Create test data
        let w_f64 = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
        ];
        let b_f64 = vec![0.1, 0.2];
        let x_f64 = vec![1.0, 2.0, 3.0];

        // Convert to field elements
        let w: Vec<Vec<Felt>> = w_f64.iter()
            .map(|row| row.iter().map(|&v| f64_to_felt(v)).collect())
            .collect();
        let b: Vec<Felt> = b_f64.iter().map(|&v| f64_to_felt(v)).collect();
        let x: Vec<Felt> = x_f64.iter().map(|&v| f64_to_felt(v)).collect();
        
        let w_sign = vec![vec![Felt::ZERO; 3]; 2];
        let b_sign = vec![Felt::ZERO; 2];
        let x_sign = vec![Felt::ZERO; 3];

        // Run our implementation
        let (result, _) = forward_propagation_layer(
            &w, &b, &x, &w_sign, &b_sign, &x_sign, f64_to_felt(1.0)
        );

        // Run reference implementation
        let expected = reference::forward_pass(&w_f64, &b_f64, &x_f64);

        // Compare results
        for (i, (&our, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            let our_f64 = felt_to_f64(our);
            assert!(compare_floats(our_f64, exp, 1e-5), 
                   "Forward pass mismatch at {}: {} vs {}", i, our_f64, exp);
        }
    }

    #[test]
    fn test_end_to_end_training_step() {
        println!("\n=== End-to-End Training Step Test ===");
        
        // Initialize small model
        let mut w_f64 = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let mut b_f64 = vec![0.1, 0.2];
        let x_f64 = vec![1.0, 0.5];
        let y_true_f64 = vec![1.0, 0.0];
        let lr = 0.1;

        // Convert to field elements
        let mut w: Vec<Vec<Felt>> = w_f64.iter()
            .map(|row| row.iter().map(|&v| f64_to_felt(v)).collect())
            .collect();
        let mut b: Vec<Felt> = b_f64.iter().map(|&v| f64_to_felt(v)).collect();
        let x: Vec<Felt> = x_f64.iter().map(|&v| f64_to_felt(v)).collect();
        let y_true: Vec<Felt> = y_true_f64.iter().map(|&v| f64_to_felt(v)).collect();

        let mut w_sign = vec![vec![Felt::ZERO; 2]; 2];
        let mut b_sign = vec![Felt::ZERO; 2];
        let x_sign = vec![Felt::ZERO; 2];

        // Forward pass (our implementation)
        println!("\n1. Forward Pass:");
        let (output, output_sign) = forward_propagation_layer(
            &w, &b, &x, &w_sign, &b_sign, &x_sign, f64_to_felt(1.0)
        );

        // Reference forward pass
        let output_ref = reference::forward_pass(&w_f64, &b_f64, &x_f64);
        println!("Our output: {:?}", output.iter().map(|&v| felt_to_f64(v)).collect::<Vec<_>>());
        println!("Expected: {:?}", output_ref);

        // Compute error (our implementation)
        println!("\n2. Error Computation:");
        let (error, error_sign) = mse_prime(&y_true, &output, &output_sign, f64_to_felt(1.0));

        // Reference error computation
        let error_ref = reference::mse_prime(&y_true_f64, &output_ref);
        println!("Our error: {:?}", error.iter().map(|&v| felt_to_f64(v)).collect::<Vec<_>>());
        println!("Expected: {:?}", error_ref);

        // Backward pass (our implementation)
        println!("\n3. Backward Pass:");
        let (new_w, new_b, _, _) = backward_propagation_layer(
            &mut w, &mut b, &x, &error, f64_to_felt(lr), f64_to_felt(1.0),
            &mut w_sign, &mut b_sign, &x_sign, &error_sign
        );

        // Reference backward pass
        reference::backward_pass(&mut w_f64, &mut b_f64, &x_f64, &error_ref, lr);
        
        println!("Our new weights: {:?}", new_w.iter().map(|row| row.iter().map(|&v| felt_to_f64(v)).collect::<Vec<_>>()).collect::<Vec<_>>());
        println!("Expected weights: {:?}", w_f64);

        // Verify results match
        for (i, (our_row, exp_row)) in new_w.iter().zip(w_f64.iter()).enumerate() {
            for (j, (&our, &exp)) in our_row.iter().zip(exp_row.iter()).enumerate() {
                let our_f64 = felt_to_f64(our);
                assert!(compare_floats(our_f64, exp, 1e-4), 
                       "End-to-end weight mismatch at [{},{}]: {} vs {}", i, j, our_f64, exp);
            }
        }

        for (i, (&our, &exp)) in new_b.iter().zip(b_f64.iter()).enumerate() {
            let our_f64 = felt_to_f64(our);
            assert!(compare_floats(our_f64, exp, 1e-4), 
                   "End-to-end bias mismatch at {}: {} vs {}", i, our_f64, exp);
        }

        println!("\n✓ End-to-end test passed!");
    }
}