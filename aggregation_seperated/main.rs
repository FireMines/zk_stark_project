use std::error::Error;
use std::time::Instant;
use helper::{f64_to_felt, generate_initial_model, read_dataset, split_dataset, AC, FE, C};
use aggregation::prover::GlobalUpdateProver;
use aggregation::air::GlobalUpdateAir;
use winterfell::crypto::hashers::Blake3_256;
use winterfell::crypto::{DefaultRandomCoin, MerkleTree};
use winterfell::{ProofOptions, Prover, Trace};
use winterfell::verify;
use winterfell::math::fields::f128::BaseElement as Felt;

mod helper;
mod aggregation {
    pub mod air;
    pub mod prover;
}

fn main() -> Result<(), Box<dyn Error>> {
    // Read the dataset from CSV (path adjusted as needed)
    let (features, labels) = read_dataset("devices/edge_device/data/train.txt")?;
    let client_data = split_dataset(features, labels, C);

    // Generate the initial global model.
    let (global_w, global_b) = generate_initial_model(FE, AC, 10000.0);

    // For each client, compute its local update L (here, by averaging its data).
    let mut local_w = Vec::new();
    let mut local_b = Vec::new();
    for (client_features, client_labels) in client_data {
        // Compute the average feature vector.
        let mut avg_feature = vec![0.0; FE];
        for row in &client_features {
            for i in 0..FE {
                avg_feature[i] += row[i];
            }
        }
        for i in 0..FE {
            avg_feature[i] /= client_features.len() as f64;
        }
        // Compute the average label.
        let sum_labels: f64 = client_labels.iter().sum();
        let avg_label = sum_labels / (client_labels.len() as f64);

        // The client’s local model update L = [L_w || L_b]
        let client_w: Vec<Vec<Felt>> = (0..AC)
            .map(|_| avg_feature.iter().map(|&x| f64_to_felt(x)).collect())
            .collect();
        let client_b: Vec<Felt> = vec![f64_to_felt(avg_label); AC];

        local_w.push(client_w);
        local_b.push(client_b);
    }

    // Scaling factor: number of clients as a field element.
    let k = f64_to_felt(C as f64);

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

    // Create the GlobalUpdateProver instance.
    let aggregator_prover = GlobalUpdateProver::new(
        proof_options.clone(),
        global_w,
        global_b,
        local_w,
        local_b,
        k,
    );

    // Build the execution trace (this includes the extra final row).
    let trace = aggregator_prover.build_trace();
    println!("Aggregator trace built with {} rows.", trace.length());

    // Retrieve public inputs.
    let pub_inputs = aggregator_prover.get_pub_inputs(&trace);
    println!("Global (old) weights: {:?}", pub_inputs.global_w);
    println!("Global (old) biases: {:?}", pub_inputs.global_b);
    println!("New Global (aggregated) weights: {:?}", pub_inputs.new_global_w);
    println!("New Global (aggregated) biases: {:?}", pub_inputs.new_global_b);
    println!("Scaling factor (k): {:?}", pub_inputs.k);
    println!("Number of update steps: {:?}", pub_inputs.steps);

    let start = Instant::now();
    let proof = aggregator_prover.prove(trace)?;
    println!(
        "Global update proof generated in {} ms",
        start.elapsed().as_millis()
    );

    // Verify the proof.
    let acceptable_options = winterfell::AcceptableOptions::OptionSet(vec![proof_options]);
    match verify::<GlobalUpdateAir, Blake3_256<Felt>, DefaultRandomCoin<Blake3_256<Felt>>, MerkleTree<Blake3_256<Felt>>>(
        proof,
        pub_inputs,
        &acceptable_options,
    ) {
        Ok(_) => println!("Global update proof verified successfully."),
        Err(e) => println!("Global update proof verification failed: {}", e),
    }

    Ok(())
}
