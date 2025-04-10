// benches/bench_mimc.rs

// Constants for MiMC benchmarks.
pub const MIMC_ROUNDS: usize = 255; // must be power of two -1, e.g. 7, 15, 31, etc.
pub const RANDOMNESS_SEED: [u8; 32] = [24u8; 32];
#[allow(dead_code)]
pub const SAMPLES: u32 = 50;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use winterfell::math::{fields::f128::BaseElement as Felt, FieldElement};
// Import your helper functions from your helper module.
use zk_stark_project::helper::{f64_to_felt, mimc_cipher, mimc_hash_matrix, get_round_constants};

/// Benchmark the MiMC cipher function.
/// It generates random field elements for the input, round constant, and uses ZERO as the chaining variable.
pub fn bench_mimc_cipher(c: &mut Criterion) {
    // Initialize a reproducible RNG using the RANDOMNESS_SEED constant.
    let mut rng: StdRng = SeedableRng::from_seed(RANDOMNESS_SEED);
    
    // Generate random values for input and round constant.
    let rand_x: u64 = rng.next_u64();
    let rand_rc: u64 = rng.next_u64();
    let x = Felt::new(rand_x as u128);
    let round_constant = Felt::new(rand_rc as u128);
    let z = Felt::ZERO; // Chaining variable

    c.bench_function("mimc_cipher", |b| {
        b.iter(|| {
            // Call mimc_cipher using black_box to prevent compiler optimizations.
            let _result = mimc_cipher(black_box(x), black_box(round_constant), black_box(z));
        })
    });
}

/// Benchmark the MiMC hash function on a weight matrix and bias vector.
/// Here we use a fixed weight matrix (with dimensions AC x FE) and bias vector (of length AC)
/// and use the round constants obtained from `get_round_constants()`.
pub fn bench_mimc_hash(c: &mut Criterion) {
    // Create a fixed weight matrix and bias vector.
    let weight_matrix: Vec<Vec<Felt>> = vec![vec![f64_to_felt(42.0); 9]; 6]; // dimensions: AC x FE
    let bias_vector: Vec<Felt> = vec![f64_to_felt(1.0); 6];                  // length: AC

    // Retrieve the round constants (here you can modify get_round_constants if you prefer to use MIMC_ROUNDS).
    let round_constants = get_round_constants();

    c.bench_function("mimc_hash_matrix", |b| {
        b.iter(|| {
            // Compute the MiMC hash over the weight matrix and bias vector.
            let _result = mimc_hash_matrix(
                black_box(&weight_matrix),
                black_box(&bias_vector),
                black_box(&round_constants),
            );
        })
    });
}

criterion_group!(benches, bench_mimc_cipher, bench_mimc_hash);
criterion_main!(benches);
