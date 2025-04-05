// src/helper.rs

use std::error::Error;
use csv::ReaderBuilder;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use winterfell::math::{FieldElement, StarkField};
use winterfell::math::fields::f128::BaseElement as Felt;

// Global constants (matching ZoKrates)
// We hard-code the values to avoid calling non-const operators.
const MAX: u128 = u128::MAX; // 340_282_366_920_938_463_463_374_607_431_768_211_455u128
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
pub fn add(a: Felt, b: Felt, a_sign: Felt, b_sign: Felt) -> (Felt, Felt) {
    let a_cleansed = if a_sign == Felt::ZERO { a } else { Felt::new(MAX) - a + Felt::ONE };
    let b_cleansed = if b_sign == Felt::ZERO { b } else { Felt::new(MAX) - b + Felt::ONE };
    let c = if a_sign == b_sign && a_sign == Felt::ONE {
        Felt::new(MAX) + Felt::ONE - a_cleansed - b_cleansed
    } else {
        a + b
    };
    let c_sign = if c.as_int() > THRESHOLD.as_int() { Felt::ONE } else { Felt::ZERO };
    (c, c_sign)
}

/// Signed subtraction with cleansing:
/// - Cleanses each operand as in `add`.
/// - If a_sign ≠ b_sign and a_sign == 0, then c = a_cleansed + b_cleansed,
///   otherwise c = a - b.
/// - The result’s sign is 1 if c > THRESHOLD, otherwise 0.
pub fn subtract(a: Felt, b: Felt, a_sign: Felt, b_sign: Felt) -> (Felt, Felt) {
    let a_cleansed = if a_sign == Felt::ZERO { a } else { Felt::new(MAX) - a + Felt::ONE };
    let b_cleansed = if b_sign == Felt::ZERO { b } else { Felt::new(MAX) - b + Felt::ONE };
    let c = if a_sign != b_sign && a_sign == Felt::ZERO {
        a_cleansed + b_cleansed
    } else {
        a - b
    };
    let d = if c.as_int() > THRESHOLD.as_int() { Felt::ONE } else { Felt::ZERO };
    (c, d)
}

/// Signed division with cleansing:
/// - Cleanses both operands.
/// - Converts them to u64 and adjusts a by subtracting its remainder modulo b.
/// - Performs the division.
/// - Sets the result’s sign to 0 if a_sign equals b_sign or the result is zero; otherwise 1.
/// - If the sign is 1, computes the final result as (MAX + 1 - res).
pub fn divide(a: Felt, b: Felt, a_sign: Felt, b_sign: Felt) -> (Felt, Felt) {
    let a_cleansed = if a_sign == Felt::ZERO { a } else { Felt::new(MAX) - a + Felt::ONE };
    let b_cleansed = if b_sign == Felt::ZERO { b } else { Felt::new(MAX) - b + Felt::ONE };

    let a_u64 = a_cleansed.as_int() as u64;
    let b_u64 = b_cleansed.as_int() as u64;
    let remainder = a_u64 % b_u64;
    let a_u64_adjusted = a_u64 - remainder;
    let a_cleansed_adjusted = u64_to_field(a_u64_adjusted);

    let res = a_cleansed_adjusted / b_cleansed;
    let sign = if a_sign == b_sign || res == Felt::ZERO { Felt::ZERO } else { Felt::ONE };
    let res_final = if sign == Felt::ZERO { res } else { Felt::new(MAX) + Felt::ONE - res };
    (res_final, sign)
}

/// Signed multiplication with cleansing:
/// - Cleanses each operand.
/// - Multiplies the cleansed operands.
/// - Sets the result’s sign to 0 if the original signs are equal or the product is zero; otherwise 1.
/// - If the sign is 1, computes the final result as (MAX - res + 1).
pub fn multiply(a: Felt, b: Felt, a_sign: Felt, b_sign: Felt) -> (Felt, Felt) {
    let a_cleansed = if a_sign == Felt::ZERO { a } else { Felt::new(MAX) - a + Felt::ONE };
    let b_cleansed = if b_sign == Felt::ZERO { b } else { Felt::new(MAX) - b + Felt::ONE };
    let res = a_cleansed * b_cleansed;
    let sign = if a_sign == b_sign || res == Felt::ZERO { Felt::ZERO } else { Felt::ONE };
    let res_final = if sign == Felt::ZERO { res } else { Felt::new(MAX) - res + Felt::ONE };
    (res_final, sign)
}


/// Reads a CSV file and extracts features and label.
/// If a row has 46 columns, it extracts columns 19–27 as features and column 46 as the label;
/// if a row has 10 columns, it treats the first 9 columns as features and the 10th as the label.
pub fn read_dataset(file_path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for result in rdr.records() {
        let record = result?;
        if record.is_empty() {
            continue;
        }
        if record.len() == 46 {
            let row: Vec<f64> = record.iter()
                .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
                .collect();
            features.push(row[18..27].to_vec());
            labels.push(row[45]);
        } else if record.len() == 10 {
            let row: Vec<f64> = record.iter()
                .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
                .collect();
            features.push(row[..9].to_vec());
            labels.push(row[9]);
        } else {
            return Err(format!("CSV row must have 10 or 46 columns, got {}", record.len()).into());
        }
    }
    Ok((features, labels))
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
pub fn generate_initial_model(input_dim: usize, output_dim: usize, precision: f64) -> (Vec<Vec<Felt>>, Vec<Felt>) {
    let normal = Normal::new(0.0, precision / 5.0).unwrap();
    let mut rng = thread_rng();
    let weights: Vec<Vec<Felt>> = (0..output_dim)
        .map(|_| {
            (0..input_dim)
                .map(|_| {
                    let sample: f64 = normal.sample(&mut rng);
                    f64_to_felt(sample)
                })
                .collect()
        })
        .collect();
    let biases: Vec<Felt> = (0..output_dim)
        .map(|_| {
            let sample: f64 = normal.sample(&mut rng);
            f64_to_felt(sample)
        })
        .collect();
    (weights, biases)
}

/// Converts a scalar label to a one‑hot vector of length AC.
pub fn label_to_one_hot(label: f64, ac: usize, precision: f64) -> Vec<Felt> {
    let mut one_hot = vec![f64_to_felt(0.0); ac];
    let idx = if label < 1.0 { 0 } else { (label as usize).saturating_sub(1) };
    if idx < ac {
        one_hot[idx] = f64_to_felt(precision);
    }
    one_hot
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
    // Determine dimensions: number of activations (ac) and features (fe).
    let ac = b.len();
    let fe = x.len();

    // --- Update the Bias Vector ---
    // For each activation unit, update the bias using:
    //   b[i] = b[i] - (output_error[i] / learning_rate)   (with sign adjustments)
    for i in 0..ac {
        // Divide the output error by the learning rate.
        let (temp, temp_sign) = divide(output_error[i], learning_rate, output_error_sign[i], Felt::ZERO);
        // Subtract this result from the bias.
        let (new_b, new_b_sign) = subtract(b[i], temp, b_sign[i], temp_sign);
        b[i] = new_b;
        b_sign[i] = new_b_sign;
    }

    // --- Update the Weight Matrix ---
    // For each feature and activation, update the weight using:
    //   w[i][j] = w[i][j] - (( (output_error[i] * x[j]) / learning_rate ) / pr)
    // with proper sign propagation.
    for j in 0..fe {
        for i in 0..ac {
            // Multiply output error and input.
            let (prod, prod_sign) = multiply(output_error[i], x[j], output_error_sign[i], x_sign[j]);
            // Divide by the learning rate.
            let (temp, temp_sign) = divide(prod, learning_rate, prod_sign, Felt::ZERO);
            // Divide the result by pr (precision/scaling).
            let (temp2, temp2_sign) = divide(temp, pr, temp_sign, Felt::ZERO);
            // Subtract the computed gradient from the current weight.
            let (new_w, new_w_sign) = subtract(w[i][j], temp2, w_sign[i][j], temp2_sign);
            w[i][j] = new_w;
            w_sign[i][j] = new_w_sign;
        }
    }

    // Return the updated weights, biases, and their corresponding sign matrices.
    (w.clone(), b.clone(), w_sign.clone(), b_sign.clone())
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