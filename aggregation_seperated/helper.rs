use std::error::Error;
use csv::ReaderBuilder;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use winterfell::math::{FieldElement, StarkField};
use winterfell::math::fields::f128::BaseElement as Felt;

// Global constants
pub const MAX: u128 = u128::MAX;
pub const AC: usize = 6; // number of activations (layers)
pub const FE: usize = 9; // number of features per activation
pub const C: usize = 8;  // number of clients
const BS: usize = 2;

/// Convert an f64 value to a field element using a scaling factor of 1e6.
pub fn f64_to_felt(x: f64) -> Felt {
    Felt::new((x * 1e6).round() as u128)
}

/// Converts a u64 value to a field element.
pub fn u64_to_field(x: u64) -> Felt {
    Felt::new(x as u128)
}

/// Converts a u32 value to a field element.
pub fn u32_to_field(x: u32) -> Felt {
    Felt::new(x as u128)
}

/// Reads a CSV file and extracts features and labels.
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

/// Splits the dataset among a given number of clients using round-robin.
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

/// Generates an initial model using a normal distribution.
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

/// Implements the MiMC cipher.  
/// For each of 64 rounds, compute: a = input + round_constant + z,
/// then input := a^7. Finally, return (input + z).
pub fn mimc_cipher(input: Felt, round_constant: Felt, z: Felt) -> Felt {
    let mut inp = input;
    for _ in 0..64 {
        let a = inp + round_constant + z;
        inp = <Felt as FieldElement>::exp(a, 7);
    }
    inp + z
}

/// Computes a MiMC hash from a weight matrix and a bias vector using provided round constants.
/// It iterates over each activation (row) and feature (column) of the weight matrix,
/// applying the MiMC cipher using a round constant (selected cyclically),
/// then similarly processes each bias.
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

/// Checks if all elements of the first slice are contained in the second slice.
/// Returns ONE if every element in `sc_lhashes` is found in `local_params_hash`, and ZERO otherwise.
pub fn local_hash_contains(sc_lhashes: &[Felt], local_params_hash: &[Felt]) -> Felt {
    let mut all_found = Felt::ONE;
    for &hash in sc_lhashes {
        let mut found = Felt::ZERO;
        for &lhash in local_params_hash {
            if hash == lhash {
                found = Felt::ONE;
                break;
            }
        }
        all_found = all_found * found;
    }
    all_found
}

/// Computes the aggregated (new) global model based on FedAvg.
/// This function implements the update:
///
///  new_global = global + (1/c) * Σᵢ (localᵢ - global)
///
/// for both weights and biases.
pub fn update_global(
    global_w: &Vec<Vec<Felt>>,
    global_b: &Vec<Felt>,
    local_w: &Vec<Vec<Vec<Felt>>>,
    local_b: &Vec<Vec<Felt>>,
) -> (Vec<Vec<Felt>>, Vec<Felt>) {
    let c = local_w.len() as u128;  // number of clients
    // Initialize new global weights as a copy of the initial global weights.
    let mut new_global_w = global_w.clone();
    let mut new_global_b = global_b.clone();

    // For each cell of the weights:
    for i in 0..global_w.len() {
        for j in 0..global_w[i].len() {
            let mut sum_diff = Felt::ZERO;
            for client in 0..local_w.len() {
                sum_diff = sum_diff + (local_w[client][i][j] - global_w[i][j]);
            }
            new_global_w[i][j] = global_w[i][j] + sum_diff / Felt::new(c);
        }
    }

    // For each bias:
    for i in 0..global_b.len() {
        let mut sum_diff = Felt::ZERO;
        for client in 0..local_b.len() {
            sum_diff = sum_diff + (local_b[client][i] - global_b[i]);
        }
        new_global_b[i] = global_b[i] + sum_diff / Felt::new(c);
    }
    (new_global_w, new_global_b)
}
