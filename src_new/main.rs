// src/main.rs

mod helper;
mod aggregation {
    pub mod air;
    pub mod prover;
}
mod training {
    pub mod air;
    pub mod prover;
}

use std::error::Error;
use std::time::Instant;
use sysinfo::{System, SystemExt};

use helper::{
    f64_to_felt, generate_initial_model, label_to_one_hot, local_hash_contains,
    mimc_hash_aggregator, mimc_hash_matrix, read_dataset, split_dataset, u32_to_field,
    AC, C, FE,
};
use training::prover::TrainingUpdateProver;
use aggregation::prover::GlobalUpdateProver;
use winterfell::crypto::hashers::Blake3_256;
use winterfell::crypto::{DefaultRandomCoin, MerkleTree};
use winterfell::math::fields::f128::BaseElement as Felt;
use winterfell::math::{FieldElement, StarkField};
use winterfell::{verify, AcceptableOptions, BatchingMethod, FieldExtension, ProofOptions, Prover, Trace};

fn main() -> Result<(), Box<dyn Error>> {
    // --- DATASET LOADING ---
    let (features, labels) = read_dataset("devices/edge_device/data/train.txt")?;
    let client_data = split_dataset(features, labels, C); // e.g. 8 clients

    // --- CLIENT SIDE: Compute local training updates.
    println!("--- Client Training Updates ---");
    let client_proof_options = ProofOptions::new(
        40,
        16,
        21,
        FieldExtension::None,
        16,
        7,
        BatchingMethod::Algebraic,
        BatchingMethod::Algebraic,
    );

    let mut total_client_time = 0;
    let mut client_final_reps = Vec::new();

    // Training phase loop over each batch (bs is implicitly defined by client_data length)
    for (i, (client_features, client_labels)) in client_data.iter().enumerate() {
        if client_features.is_empty() {
            return Err("Client data is empty".into());
        }
        let sample = &client_features[0];
        if sample.len() != FE {
            return Err(format!(
                "CSV row must have {} feature columns, got {}",
                FE,
                sample.len()
            )
            .into());
        }
        let x: Vec<Felt> = sample.iter().map(|&v| f64_to_felt(v)).collect();
        let x_sign: Vec<Felt> = vec![f64_to_felt(0.0); x.len()];
        let label_val = client_labels[0];
        let y: Vec<Felt> = label_to_one_hot(label_val, AC, 1e6);
        let (init_w, init_b) = generate_initial_model(FE, AC, 10000.0);
        let learning_rate = f64_to_felt(0.0001);
        let precision = f64_to_felt(1e6);
        // For training the prover, initialize sign vectors (we assume default 0).
        let w_sign: Vec<Vec<Felt>> = vec![vec![f64_to_felt(0.0); FE]; AC];
        let b_sign: Vec<Felt> = vec![f64_to_felt(0.0); AC];

        let start = Instant::now();

        let training_prover = TrainingUpdateProver::new(
            client_proof_options.clone(),
            init_w.clone(),
            init_b.clone(),
            w_sign,
            b_sign,
            x,
            x_sign,
            y,
            learning_rate,
            precision,
        );
        let trace = training_prover.build_trace();

        // --- PROOF GENERATION AND BENCHMARKS FOR CLIENT TRAINING ---
        let training_proof = training_prover.prove(trace.clone())?;
        let proof_time = start.elapsed();
        println!("Client {}: Proof generation time: {} ms", i + 1, proof_time.as_millis());

        let proof_bytes = training_proof.to_bytes();
        println!("Client {}: Training proof size: {} bytes", i + 1, proof_bytes.len());

        // Memory usage via sysinfo.
        let mut sys = System::new_all();
        sys.refresh_all();
        println!("Client {}: Memory usage: {} KB", i + 1, sys.used_memory() as f64 / 1024.0);

        // --- VERIFY THE TRAINING PROOF BEFORE MOVING ON ---
        let training_pub_inputs = training_prover.get_pub_inputs(&trace);
        match verify::<training::air::TrainingUpdateAir, Blake3_256<Felt>, DefaultRandomCoin<Blake3_256<Felt>>, MerkleTree<Blake3_256<Felt>>>(
            training_proof,
            training_pub_inputs,
            &AcceptableOptions::OptionSet(vec![client_proof_options.clone()]),
        ) {
            Ok(_) => println!("Training proof for client {} verified successfully.", i + 1),
            Err(e) => println!("Training proof verification for client {} failed: {}", i + 1, e),
        }
        let elapsed = start.elapsed().as_millis();
        println!("Client {}: Total proof generation time: {} ms\n", i + 1, elapsed);
        total_client_time += elapsed;

        let pub_inputs = training_prover.get_pub_inputs(&trace);
        client_final_reps.push(pub_inputs.final_state[0]);
    }
    println!(
        "Average client update time: {} ms",
        total_client_time / (client_data.len() as u128)
    );

    // --- GLOBAL UPDATE: FedAvg Aggregation ---
    println!("\n--- Global Update Example ---");
    let (global_w, global_b) = generate_initial_model(FE, AC, 10000.0);
    let global_w_sign: Vec<Vec<Felt>> = vec![vec![f64_to_felt(0.0); FE]; AC];
    let global_b_sign: Vec<Felt> = vec![f64_to_felt(0.0); AC];

    let mut local_w = Vec::new();
    let mut local_w_sign = Vec::new();
    let mut local_b = Vec::new();
    let mut local_b_sign = Vec::new();
    for rep in client_final_reps.iter() {
        let client_val = (*rep).as_int() as f64 / 1e6;
        let client_w_mat: Vec<Vec<Felt>> = vec![vec![f64_to_felt(client_val); FE]; AC];
        let client_w_sign_mat: Vec<Vec<Felt>> = client_w_mat
            .iter()
            .map(|row| row.iter().map(|_| f64_to_felt(0.0)).collect())
            .collect();
        let client_b_vec: Vec<Felt> = vec![f64_to_felt(client_val); AC];
        let client_b_sign_vec: Vec<Felt> =
            client_b_vec.iter().map(|_| f64_to_felt(0.0)).collect();
        local_w.push(client_w_mat);
        local_w_sign.push(client_w_sign_mat);
        local_b.push(client_b_vec);
        local_b_sign.push(client_b_sign_vec);
    }

    // Clone local parameters for later local hash checking.
    let local_w_clone = local_w.clone();
    let local_b_clone = local_b.clone();

    let k = f64_to_felt(C as f64);
    let aggregator_prover = GlobalUpdateProver::new(
        client_proof_options.clone(),
        global_w,
        global_w_sign,
        global_b,
        global_b_sign,
        local_w,
        local_w_sign,
        local_b,
        local_b_sign,
        k,
    );
    let trace = aggregator_prover.build_trace();
    println!("Aggregator trace built with {} rows.", trace.length());
    let trace_rows = aggregator_prover.compute_iterative_trace();
    let final_state = trace_rows.last().unwrap().clone();
    let new_global_w: Vec<Vec<Felt>> = final_state[..(AC * FE)]
        .chunks(FE)
        .map(|chunk| chunk.to_vec())
        .collect();
    let new_global_b: Vec<Felt> = final_state[(AC * FE)..].to_vec();

    // Define round constants as a fixed-size array.
    let round_constants: [Felt; 64] = [
        f64_to_felt(42.0),
        f64_to_felt(43.0),
        f64_to_felt(170.0),
        f64_to_felt(2209.0),
        f64_to_felt(16426.0),
        f64_to_felt(78087.0),
        f64_to_felt(279978.0),
        f64_to_felt(823517.0),
        f64_to_felt(2097194.0),
        f64_to_felt(4782931.0),
        f64_to_felt(10000042.0),
        f64_to_felt(19487209.0),
        f64_to_felt(35831850.0),
        f64_to_felt(62748495.0),
        f64_to_felt(105413546.0),
        f64_to_felt(170859333.0),
        f64_to_felt(268435498.0),
        f64_to_felt(410338651.0),
        f64_to_felt(612220074.0),
        f64_to_felt(893871697.0),
        f64_to_felt(1280000042.0),
        f64_to_felt(1801088567.0),
        f64_to_felt(2494357930.0),
        f64_to_felt(3404825421.0),
        f64_to_felt(4586471466.0),
        f64_to_felt(6103515587.0),
        f64_to_felt(8031810218.0),
        f64_to_felt(10460353177.0),
        f64_to_felt(13492928554.0),
        f64_to_felt(17249876351.0),
        f64_to_felt(21870000042.0),
        f64_to_felt(27512614133.0),
        f64_to_felt(34359738410.0),
        f64_to_felt(42618442955.0),
        f64_to_felt(52523350186.0),
        f64_to_felt(64339296833.0),
        f64_to_felt(78364164138.0),
        f64_to_felt(94931877159.0),
        f64_to_felt(114415582634.0),
        f64_to_felt(137231006717.0),
        f64_to_felt(163840000042.0),
        f64_to_felt(194754273907.0),
        f64_to_felt(230539333290.0),
        f64_to_felt(271818611081.0),
        f64_to_felt(319277809706.0),
        f64_to_felt(373669453167.0),
        f64_to_felt(435817657258.0),
        f64_to_felt(506623120485.0),
        f64_to_felt(587068342314.0),
        f64_to_felt(678223072891.0),
        f64_to_felt(781250000042.0),
        f64_to_felt(897410677873.0),
        f64_to_felt(1028071702570.0),
        f64_to_felt(1174711139799.0),
        f64_to_felt(1338925210026.0),
        f64_to_felt(1522435234413.0),
        f64_to_felt(1727094849578.0),
        f64_to_felt(1954897493219.0),
        f64_to_felt(2207984167594.0),
        f64_to_felt(2488651484857.0),
        f64_to_felt(2799360000042.0),
        f64_to_felt(3142742835999.0),
        f64_to_felt(3521614606250.0),
        f64_to_felt(3938980639125.0),
    ];

    // Compute the MiMC digest for the new global parameters.
    let computed_digest = mimc_hash_matrix(&new_global_w, &new_global_b, &round_constants);
    let mut pub_inputs = aggregator_prover.get_pub_inputs(&trace);
    pub_inputs.digest = computed_digest;
    println!("Global (old) weights: {:?}", pub_inputs.global_w);
    println!("Global (old) biases:  {:?}", pub_inputs.global_b);
    println!("New Global (aggregated) weights: {:?}", pub_inputs.new_global_w);
    println!("New Global (aggregated) biases:  {:?}", pub_inputs.new_global_b);
    println!("Computed MiMC digest: {:?}", computed_digest);

    // Clone values needed for later verification.
    let expected_digest = pub_inputs.digest.clone();
    let expected_new_global_w = pub_inputs.new_global_w.clone();
    let expected_new_global_b = pub_inputs.new_global_b.clone();

    let start = Instant::now();
    let proof = aggregator_prover.prove(trace)?;
    println!(
        "Global update proof generated in {} ms",
        start.elapsed().as_millis()
    );
    // Uncomment the line below to print the proof in hex.
    // println!("Global update proof (hex): {}", hex::encode(proof.to_bytes()));
    let acceptable_options = AcceptableOptions::OptionSet(vec![client_proof_options]);
    match verify::<aggregation::air::GlobalUpdateAir, Blake3_256<Felt>, DefaultRandomCoin<Blake3_256<Felt>>, MerkleTree<Blake3_256<Felt>>>(
        proof,
        pub_inputs,
        &acceptable_options,
    ) {
        Ok(_) => println!("Global update proof verified successfully."),
        Err(e) => println!("Global update proof verification failed: {}", e),
    }
    
    // --- Check local model hashes via aggregator ---
    // Here we compute a hash for each client's local parameters and then check that against
    // the aggregator's stored local hashes.
    // Compute sc_lhashes (the aggregator's stored local hashes) from local parameters.
    let mut sc_lhashes = Vec::with_capacity(C);
    for i in 0..C {
        let hash = mimc_hash_aggregator(&local_w_clone[i], &local_b_clone[i], &round_constants);
        sc_lhashes.push(hash);
    }
    // Also compute local_hash_check from the same local parameters.
    let mut local_hash_check = vec![Felt::ZERO; C];
    for i in 0..C {
        local_hash_check[i] = mimc_hash_aggregator(&local_w_clone[i], &local_b_clone[i], &round_constants);
    }
    let lhashes_match = local_hash_contains(&sc_lhashes, &local_hash_check);
    assert!(lhashes_match == Felt::ONE);

    // Finally, compare computed global hash with expected digest.
    let result = if mimc_hash_aggregator(&expected_new_global_w, &expected_new_global_b, &round_constants)
        == expected_digest
    {
        Felt::ONE
    } else {
        Felt::ZERO
    };
    if result == Felt::ONE {
        println!("Final result: global model aggregation hash matches expected digest.");
    } else {
        println!("Final result: global model aggregation hash does not match expected digest.");
    }

    Ok(())
}
