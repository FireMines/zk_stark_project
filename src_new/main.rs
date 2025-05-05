// src/main.rs

mod helper;
mod signed;
mod training {
    pub mod air;
    pub mod prover;
}
mod aggregation {
    pub mod air;
    pub mod prover;
}

use std::error::Error;
use std::time::Instant;
use std::str::FromStr;

use structopt::StructOpt;

use helper::{
    f64_to_felt, flatten_state_matrix, generate_initial_model, label_to_one_hot, 
    read_dataset, split_dataset, split_state_with_sign, AC, C, FE
};
use training::prover::TrainingUpdateProver;
use training::air::TrainingUpdateAir;
use aggregation::prover::GlobalUpdateProver;
use aggregation::air::GlobalUpdateAir;

use winterfell::math::{FieldElement, StarkField};
use winterfell::{
    verify, AcceptableOptions, BatchingMethod, FieldExtension, ProofOptions, Prover, Trace
};
use winterfell::crypto::{DefaultRandomCoin, MerkleTree, hashers::Blake3_256};
use winterfell::math::fields::f128::BaseElement as Felt;

/// Which step to run: setup, witness, or proof
#[derive(Debug, Clone)]
enum Step {
    Setup,
    Witness,
    Proof,
}
impl FromStr for Step {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "setup"   => Ok(Step::Setup),
            "witness" => Ok(Step::Witness),
            "proof"   => Ok(Step::Proof),
            other     => Err(format!("Unknown step: {}", other)),
        }
    }
}

/// CLI args
#[derive(Debug, StructOpt)]
#[structopt(name="stark_aggregator", about="STARK Aggregator with built‑in training")]
struct Cli {
    /// step to run
    #[structopt(long, default_value="setup")]
    step: Step,
    /// how many clients (overrides helper::C)
    #[structopt(long, default_value="8")]
    clients: usize,
    /// local SGD batch size (number of samples per batch)
    #[structopt(long, default_value="2")]
    batch_size: usize,
    /// maximum batches to process per client (for demo/testing)
    #[structopt(long, default_value="3")]
    max_batches: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::from_args();
    let overall_start = Instant::now();

    println!("Starting program with {} clients, batch size {}, max batches {}", 
             args.clients, args.batch_size, args.max_batches);

    //------------------------------------------------------------
    // 1) Common proof options
    //------------------------------------------------------------
    let proof_options = ProofOptions::new(
        40,      // TRACE‑layout (log2)
        16,      // FRI lde blowup
        21,      // FRI query count
        FieldExtension::None,
        16,      // FRI lde folds
        7,       // FRI cap
        BatchingMethod::Algebraic,
        BatchingMethod::Algebraic,
    );

    //------------------------------------------------------------
    // 2) Load & split dataset
    //------------------------------------------------------------
    println!("Loading dataset...");
    let (features, labels) = read_dataset("devices/edge_device/data/train.txt")?;
    let client_data = split_dataset(features, labels, args.clients);
    println!("Dataset loaded and split among {} clients", args.clients);

    //------------------------------------------------------------
    // 3) Run the training update for each client
    //------------------------------------------------------------
    println!("--- Client Training Updates ---");
    let mut client_reps = Vec::with_capacity(args.clients);

    for (i, (client_feats, client_labs)) in client_data.iter().enumerate() {
        println!("Processing client {}/{}", i+1, args.clients);
        
        // Start from a fresh random init per client
        let (mut w, mut w_sign, mut b, mut b_sign) =
            generate_initial_model(FE, AC, /*σ=*/1.0);

        let lr = f64_to_felt(0.0001); // Scale and cast to u128
        let pr = f64_to_felt(1e6);

        // Create mini-batches of size `batch_size`
        let mut batches = Vec::new();
        
        // Limit the number of samples we process for demonstration
        let total_samples = client_feats.len().min(100); // Process max 100 samples
        
        // Create mini-batches from the samples
        for batch_start in (0..total_samples).step_by(args.batch_size) {
            let batch_end = (batch_start + args.batch_size).min(total_samples);
            
            // If we have at least one full sample, add the batch
            if batch_end > batch_start {
                let feat_batch = &client_feats[batch_start..batch_end];
                let lab_batch = &client_labs[batch_start..batch_end];
                batches.push((feat_batch, lab_batch));
            }
        }
            
        println!("Client {}: Processing {} mini-batches (max {})", 
                 i+1, batches.len().min(args.max_batches), args.max_batches);

        for (batch_idx, (feat_batch, lab_batch)) in batches.iter().enumerate() {
            // Only process up to max_batches
            if batch_idx >= args.max_batches {
                println!("Reached maximum batches limit ({})", args.max_batches);
                break;
            }
            
            println!("Client {}: Processing batch {}/{}", 
                     i+1, batch_idx+1, batches.len().min(args.max_batches));
            
            let B = feat_batch.len();
            println!("  Batch size: {}", B);

            // 3a) Flatten this batch's features & labels
            let mut x_vals  = Vec::with_capacity(B * FE);
            let mut x_signs = Vec::with_capacity(B * FE);
            let mut y_onehot = Vec::with_capacity(B * AC);
            let mut y_sign   = Vec::with_capacity(B * AC);

            for (feat_row, &lab) in feat_batch.iter().zip(lab_batch.iter()) {
                // feat_row: &Vec<f64>, lab: f64
                x_vals.extend(feat_row.iter().map(|&v| f64_to_felt(v)));
                x_signs.extend(std::iter::repeat(Felt::ZERO).take(FE));
                let (yh, ys) = label_to_one_hot(lab, AC, /*precision=*/1e6);
                y_onehot.extend(yh);
                y_sign.extend(ys);
            }

            // 3b) Build & prove the STARK on the current (w,b) and this batch
            println!("  Creating prover for client {} batch {}", i+1, batch_idx+1);
            let prover = TrainingUpdateProver::new(
                proof_options.clone(),
                w.clone(),           // current weights
                b.clone(),           // current biases
                w_sign.clone(),
                b_sign.clone(),
                x_vals.clone(),
                x_signs.clone(),
                y_onehot.clone(),
                y_sign.clone(),
                lr,
                pr,
            );

            println!("  Building trace for client {} batch {}", i+1, batch_idx+1);
            let trace = prover.build_trace();
            
            println!("  Generating proof for client {} batch {}", i+1, batch_idx+1);
            let proof = match prover.prove(trace.clone()) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Error generating proof for client {} batch {}: {:?}", i+1, batch_idx+1, e);
                    return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, 
                        format!("Proof generation error: {:?}", e))));
                }
            };
            
            println!("  Verifying proof for client {} batch {}", i+1, batch_idx+1);
            match verify::<
                TrainingUpdateAir, Blake3_256<Felt>,
                DefaultRandomCoin<Blake3_256<Felt>>, MerkleTree<Blake3_256<Felt>>
            >(proof, prover.get_pub_inputs(&trace), 
            &AcceptableOptions::OptionSet(vec![proof_options.clone()])) {
                Ok(_) => println!("  Proof verification successful for client {} batch {}", i+1, batch_idx+1),
                Err(e) => {
                    eprintln!("Error verifying proof for client {} batch {}: {:?}", i+1, batch_idx+1, e);
                    return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, 
                        format!("Proof verification error: {:?}", e))));
                }
            }

            // 3c) Extract the updated (w,b) from the last trace row
            let last_row = trace.length() - 1;
            println!("  Extracting updated model from trace row {}", last_row);

            // Extract the masked state from the last row
            let mut masked_state = Vec::with_capacity(trace.width());
            for col in 0..trace.width() {
                masked_state.push(trace.get(col, last_row));
            }
            
            // For masked state, we need to extract the actual values
            // In our trace format, for each parameter v with mask m, we store (v+m, m)
            // To recover v, we compute (v+m) - m

            // Calculate the number of parameters
            let num_weights = AC * FE;
            let num_biases = AC;
            let num_params = num_weights + num_biases;
            
            // Initialize arrays for unmasked values
            let mut unmasked_state = Vec::with_capacity(num_params);
            
            // Extract unmasked values: for each pair (v+m, m), compute (v+m) - m = v
            for i in (0..masked_state.len()).step_by(2) {
                if i + 1 < masked_state.len() {
                    let masked_value = masked_state[i];
                    let mask = masked_state[i + 1];
                    unmasked_state.push(masked_value - mask);
                }
            }
            
            // Extract weights and biases
            let mut param_idx = 0;
            
            // Extract weights
            for i in 0..AC {
                for j in 0..FE {
                    if param_idx < unmasked_state.len() {
                        w[i][j] = unmasked_state[param_idx];
                        param_idx += 1;
                    }
                }
            }
            
            // Extract biases
            for i in 0..AC {
                if param_idx < unmasked_state.len() {
                    b[i] = unmasked_state[param_idx];
                    param_idx += 1;
                }
            }
            
            // Extract weight signs
            for i in 0..AC {
                for j in 0..FE {
                    if param_idx < unmasked_state.len() {
                        w_sign[i][j] = unmasked_state[param_idx];
                        param_idx += 1;
                    }
                }
            }
            
            // Extract bias signs
            for i in 0..AC {
                if param_idx < unmasked_state.len() {
                    b_sign[i] = unmasked_state[param_idx];
                    param_idx += 1;
                }
            }
            
            println!("  Completed client {} batch {}/{}", i+1, batch_idx+1, batches.len().min(args.max_batches));
        }

        // After all mini‐batches, flatten the final model into a single vector
        println!("Client {}: Training complete, flattening model", i+1);
        client_reps.push(flatten_state_matrix(&w, &b));
        println!("Client {}/{} processing complete", i+1, args.clients);
    }
    
    println!("All client training updates complete");
    
    //------------------------------------------------------------
    // 4) Build local_w, local_b from those scalars
    //------------------------------------------------------------
    println!("Building local models from client representations...");
    let mut local_w = Vec::with_capacity(args.clients);
    let mut local_b = Vec::with_capacity(args.clients);
    for (i, rep) in client_reps.iter().enumerate() {
        println!("Processing client representation {}/{}", i+1, client_reps.len());
        let val = rep.iter().map(|elem| elem.as_int() as f64).sum::<f64>() / (rep.len() as f64 * 1e6);
        let w_mat = vec![vec![f64_to_felt(val); FE]; AC];
        let b_vec = vec![f64_to_felt(val); AC];
        local_w.push(w_mat);
        local_b.push(b_vec);
    }

    //------------------------------------------------------------
    // 5) Initialize global model
    //------------------------------------------------------------
    println!("Initializing global model...");
    let (raw_global_w, gw_sign, raw_global_b, gb_sign) =
        generate_initial_model(FE, AC, /*σ=*/10_000.0);
    let k = f64_to_felt(args.clients as f64);

    //------------------------------------------------------------
    // 6) Create aggregator prover once
    //------------------------------------------------------------
    println!("Creating aggregator prover...");
    let agg0 = Instant::now();
    let aggregator_prover = GlobalUpdateProver::new(
        proof_options.clone(),
        raw_global_w.clone(),
        raw_global_b.clone(),
        local_w.clone(),
        local_b.clone(),
        k,
    );
    println!(
        "Aggregator prover instantiated in {} ms\n",
        agg0.elapsed().as_millis()
    );

    //------------------------------------------------------------
    // 7) Now dispatch on --step
    //------------------------------------------------------------
    match args.step {
        Step::Setup => {
            println!("STEP=setup: data+training+aggregator setup complete, no trace/proof built.");
        }

        Step::Witness => {
            println!("STEP=witness: building aggregator trace only…");
            let t0 = Instant::now();
            let trace = aggregator_prover.build_trace();
            println!(
                "  ↳ built trace with {} rows in {} ms",
                trace.length(),
                t0.elapsed().as_millis()
            );
        }

        Step::Proof => {
            println!("STEP=proof: build trace, generate + verify aggregator proof…");

            let t1 = Instant::now();
            println!("Building aggregator trace...");
            let trace = aggregator_prover.build_trace();
            println!(
                "  ↳ trace with {} rows in {} ms",
                trace.length(),
                t1.elapsed().as_millis()
            );

            let t2 = Instant::now();
            println!("Generating aggregator proof...");
            let proof = match aggregator_prover.prove(trace.clone()) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Error generating aggregator proof: {:?}", e);
                    return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, 
                        format!("Aggregator proof generation error: {:?}", e))));
                }
            };
            println!(
                "  ↳ proof generated in {} ms, size = {} bytes",
                t2.elapsed().as_millis(),
                proof.to_bytes().len()
            );

            print!("  ↳ verifying… ");
            let pub_inputs = aggregator_prover.get_pub_inputs(&trace);
            match verify::<
                GlobalUpdateAir,
                Blake3_256<Felt>,
                DefaultRandomCoin<Blake3_256<Felt>>,
                MerkleTree<Blake3_256<Felt>>
            >(proof, pub_inputs, &AcceptableOptions::OptionSet(vec![proof_options.clone()])) {
                Ok(_) => println!("OK"),
                Err(e) => {
                    eprintln!("Error verifying aggregator proof: {:?}", e);
                    return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, 
                        format!("Aggregator proof verification error: {:?}", e))));
                }
            }
        }
    }

    println!(
        "\nOverall run time: {} ms",
        overall_start.elapsed().as_millis()
    );
    Ok(())
}