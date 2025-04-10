// aggregation_seperated/helper.rs

use std::error::Error;
use csv::ReaderBuilder;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use winterfell::math::fields::f128::BaseElement as Felt;
use winterfell::math::FieldElement;

pub const AC: usize = 6; // number of activations
pub const FE: usize = 9; // features per activation
pub const C: usize = 8;  // number of clients

/// Convert an f64 value to a field element using a scaling factor of 1e6.
pub fn f64_to_felt(x: f64) -> Felt {
    Felt::new((x * 1e6).round() as u128)
}

/// Reads a CSV file and extracts features + labels.
pub fn read_dataset(file_path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for result in rdr.records() {
        let record = result?;
        if record.len() == 46 {
            let row: Vec<f64> = record.iter().map(|s| s.trim().parse::<f64>().unwrap_or(0.0)).collect();
            features.push(row[18..27].to_vec());
            labels.push(row[45]);
        } else if record.len() == 10 {
            let row: Vec<f64> = record.iter().map(|s| s.trim().parse::<f64>().unwrap_or(0.0)).collect();
            features.push(row[..9].to_vec());
            labels.push(row[9]);
        }
    }
    Ok((features, labels))
}

/// Splits data among c clients via round-robin.
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
pub fn generate_initial_model(input_dim: usize, output_dim: usize, precision: f64)
-> (Vec<Vec<Felt>>, Vec<Felt>) {
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

/// A direct aggregator function for FedAvg if needed off-chain (unmasked).
pub fn update_global_iterative(
    global_w: &Vec<Vec<Felt>>,
    global_b: &Vec<Felt>,
    local_w: &Vec<Vec<Vec<Felt>>>,
    local_b: &Vec<Vec<Felt>>,
) -> (Vec<Vec<Felt>>, Vec<Felt>) {
    // S_new = S_0 + (1/c)*∑(L_i - S_0)
    // We do it iteratively: each update uses base S_0.
    let c = local_w.len() as u128;
    let divisor = Felt::new(c);

    let mut state_w = global_w.clone();
    let mut state_b = global_b.clone();

    for client in 0..local_w.len() {
        for i in 0..state_w.len() {
            for j in 0..state_w[i].len() {
                let delta = (local_w[client][i][j] - global_w[i][j]) / divisor;
                state_w[i][j] = state_w[i][j] + delta;
            }
        }
        for i in 0..state_b.len() {
            let delta = (local_b[client][i] - global_b[i]) / divisor;
            state_b[i] = state_b[i] + delta;
        }
    }
    (state_w, state_b)
}

/// Additional helper: transpose for building the trace
pub fn transpose(matrix: Vec<Vec<Felt>>) -> Vec<Vec<Felt>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![Felt::ZERO; rows]; cols];
    for i in 0..rows {
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

pub fn get_round_constants() -> Vec<Felt> {
    (1..=64).map(|i| f64_to_felt(i as f64)).collect()
}

pub fn mimc_hash_matrix(w: &[Vec<Felt>], b: &[Felt], round_constants: &[Felt]) -> Felt {
    let mut z = Felt::ZERO;
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

/// Check local hashes
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
