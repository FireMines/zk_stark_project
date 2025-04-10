use std::error::Error;
use std::time::Instant;
use winterfell::math::ToElements;
use zk_stark_project::helper::{
    self, f64_to_felt, generate_initial_model, read_dataset, split_dataset,
};
use zk_stark_project::aggregation::air::GlobalUpdateAir;
use zk_stark_project::aggregation::prover::GlobalUpdateProver; // <== This is your masked prover
use winterfell::crypto::hashers::Blake3_256;
use winterfell::crypto::{DefaultRandomCoin, MerkleTree};
use winterfell::{ProofOptions, Prover, Trace, verify};
use winterfell::math::fields::f128::BaseElement as Felt;

fn main() -> Result<(), Box<dyn Error>> {
    // Track overall time.
    let overall_start = Instant::now();

    // --- Setup phase ---
    let setup_start = Instant::now();

    // Read dataset (path assumed correct)
    let (features, labels) = read_dataset("devices/edge_device/data/train.txt")?;
    let client_data = split_dataset(features, labels, helper::C);

    // Generate the *real* (unmasked) global model.
    // The prover will internally create a masked version.
    let (raw_global_w, raw_global_b) = generate_initial_model(helper::FE, helper::AC, 10000.0);

    // Compute client updates (simple averaging).
    let mut local_w = Vec::new();
    let mut local_b = Vec::new();
    for (client_features, client_labels) in client_data {
        let mut avg_feature = vec![0.0; helper::FE];
        for row in &client_features {
            for i in 0..helper::FE {
                avg_feature[i] += row[i];
            }
        }
        for i in 0..helper::FE {
            avg_feature[i] /= client_features.len() as f64;
        }
        let sum_labels: f64 = client_labels.iter().sum();
        let avg_label = sum_labels / (client_labels.len() as f64);
        let client_w: Vec<Vec<Felt>> = (0..helper::AC)
            .map(|_| avg_feature.iter().map(|&x| f64_to_felt(x)).collect())
            .collect();
        let client_b: Vec<Felt> = vec![f64_to_felt(avg_label); helper::AC];
        local_w.push(client_w);
        local_b.push(client_b);
    }

    // Scaling factor: number of clients (scaled by 1e6).
    let k = f64_to_felt(helper::C as f64);

    // Set proof options.
    let proof_options = ProofOptions::new(
        40,
        16,
        21,
        winterfell::FieldExtension::None,
        16,
        7,
        winterfell::BatchingMethod::Algebraic,
        winterfell::BatchingMethod::Algebraic,
    );

    let setup_time = setup_start.elapsed();
    println!("Setup time: {} ms", setup_time.as_millis());

    // --- Prover creation and trace generation ---
    // Here we pass the *unmasked* raw global model to the prover. The prover will create
    // a masked version internally for row0. The entire trace is masked -> fully ZK.
    let prover_start = Instant::now();
    let aggregator_prover = GlobalUpdateProver::new(
        proof_options.clone(),
        raw_global_w.clone(), // real old weights
        raw_global_b.clone(), // real old biases
        local_w.clone(),
        local_b.clone(),
        k,
    );
    let trace = aggregator_prover.build_trace();
    let trace_time = prover_start.elapsed();
    println!("STARK trace built with {} rows.", trace.length());
    println!("Prover & trace generation time: {} ms", trace_time.as_millis());

    // --- Get public inputs from the STARK trace ---
    // These "public inputs" are masked old + masked new states
    let pub_inputs = aggregator_prover.get_pub_inputs(&trace);
    /* 
    println!("Masked old global weights: {:?}", pub_inputs.global_w);
    println!("Masked old global biases: {:?}", pub_inputs.global_b);
    println!("Masked new global weights: {:?}", pub_inputs.new_global_w);
    println!("Masked new global biases: {:?}", pub_inputs.new_global_b);
    println!("Scaling factor (k): {:?}", pub_inputs.k);
    println!("Number of update steps: {:?}", pub_inputs.steps);
    println!("On-chain computed digest (masked final): {:?}", pub_inputs.digest);
    */
    // --- Proof generation ---
    let proof_gen_start = Instant::now();
    let proof = aggregator_prover.prove(trace)?;
    let proof_gen_time = proof_gen_start.elapsed();
    println!("Global update proof generated in {} ms", proof_gen_time.as_millis());

    // --- Proof statistics ---
    let runtime_proof_size = std::mem::size_of_val(&proof);
    let serialized_proof = proof.to_bytes();
    println!("Runtime proof size (stack): {} bytes", runtime_proof_size);
    println!("Serialized proof size: {} bytes", serialized_proof.len());

    // Public witness size: number of field elements in public inputs (all masked).
    let pub_witness = pub_inputs.to_elements();
    println!("Public witness size: {} elements", pub_witness.len());

    // --- Proof verification ---
    let verify_start = Instant::now();
    let acceptable_options = winterfell::AcceptableOptions::OptionSet(vec![proof_options]);
    match verify::<GlobalUpdateAir, Blake3_256<Felt>, DefaultRandomCoin<Blake3_256<Felt>>, MerkleTree<Blake3_256<Felt>>>(
        proof.clone(),
        pub_inputs.clone(),
        &acceptable_options,
    ) {
        Ok(_) => println!("Global update proof verified successfully (entirely masked)."),
        Err(e) => println!("Global update proof verification failed: {}", e),
    }
    let verify_time = verify_start.elapsed();
    println!("Proof verification time: {} ms", verify_time.as_millis());

    /*
    // --- Off-chain aggregation (iterative) ---
    // The aggregator can compute the final real model using update_global_iterative if needed:
    let offchain_start = Instant::now();
    let (offchain_w, offchain_b) = update_global_iterative(&raw_global_w, &raw_global_b, &local_w, &local_b);
    let offchain_time = offchain_start.elapsed();
    println!("Off-chain aggregation time: {} ms", offchain_time.as_millis());

    let round_constants = get_round_constants();
    let offchain_digest = mimc_hash_matrix(&offchain_w, &offchain_b, &round_constants);
    println!("Off-chain *unmasked* final global weights: {:?}", offchain_w);
    println!("Off-chain *unmasked* final global biases: {:?}", offchain_b);
    println!("Off-chain computed digest: {:?}", offchain_digest);

    // For each client, compute local hash.
    let mut local_hashes = Vec::new();
    for client in 0..local_w.len() {
        let hash = mimc_hash_matrix(&local_w[client], &local_b[client], &round_constants);
        local_hashes.push(hash);
    }
    let check = local_hash_contains(&local_hashes, &local_hashes);
    println!("Local hash check (should be ONE): {:?}", check);
    */
    let overall_time = overall_start.elapsed();
    println!("Overall execution time: {} ms", overall_time.as_millis());

    Ok(())
}
