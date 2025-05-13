// src/main.rs

pub mod signed;
mod helper;
mod debug; 
mod training {
    pub mod air;
    pub mod prover;
}
mod aggregation {
    pub mod air;
    pub mod prover;
}

use std::{error::Error, fs, path::PathBuf, str::FromStr, time::Instant};
use structopt::StructOpt;

use helper::{
    EdgeDevice, read_dataset,
    f64_to_felt, generate_initial_model, label_to_one_hot,
    AC, FE,
};
use training::prover::TrainingUpdateProver;
use training::air::TrainingUpdateAir;
use aggregation::prover::GlobalUpdateProver;
use aggregation::air::GlobalUpdateAir;

use winterfell::math::{FieldElement, StarkField};
use winterfell::{
    verify, AcceptableOptions, BatchingMethod,
    FieldExtension, ProofOptions, Prover, Trace,
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
#[structopt(name = "zk_stark_project", about = "STARK Aggregator with built-in training")]
struct Cli {
    /// step to run: setup, witness, or proof
    #[structopt(long, default_value = "setup")]
    step: Step,

    /// path to folder containing Device_*/data files
    #[structopt(long, default_value = "devices/edge_device/data")]
    data_dir: String,

    /// batch size (`bs`) for the ZK circuit - how many samples per proof
    #[structopt(long, default_value = "1")]
    bs: usize,
    
    /// Enable verbose output for benchmarking
    #[structopt(long)]
    verbose: bool,
}

// host-side sample size (mirrors Python's NumberOfSamplesGenerated)
const SAMPLE_SIZE: usize = 50;

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::from_args();
    let overall_start = Instant::now();

    // Ensure batch size is positive and reasonable
    if args.bs == 0 {
        return Err("ZK circuit batch size must be positive".into());
    }
    if args.bs > SAMPLE_SIZE {
        return Err(format!("ZK circuit batch size ({}) cannot exceed sample size ({})", args.bs, SAMPLE_SIZE).into());
    }

    // DEBUG: Add explicit logging for batch size
    println!("DEBUG: Starting with batch size = {}", args.bs);
    println!("DEBUG: Step = {:?}", args.step);

    //------------------------------------------------------------
    // 1) Common STARK proof options
    //------------------------------------------------------------
    let proof_options = ProofOptions::new(
        40,   // TRACE-layout (log2)
        16,   // FRI blowup
        21,   // FRI query count
        FieldExtension::None,
        16,   // FRI folds
        7,    // FRI cap
        BatchingMethod::Algebraic,
        BatchingMethod::Algebraic,
    );

    //------------------------------------------------------------
    // 2) Discover Device_* folders & load each as a client
    //------------------------------------------------------------
    let mut devices = Vec::new();
    for entry in fs::read_dir(&args.data_dir)? {
        let path = entry?.path();
        let name = path.file_name().unwrap().to_string_lossy();
        if !path.is_dir() || !name.starts_with("Device_") {
            continue;
        }
        // look for train.txt or device_data.txt
        let mut ds = PathBuf::from(&path);
        ds.push("train.txt");
        if !ds.exists() {
            ds.pop();
            ds.push("device_data.txt");
        }
        if !ds.exists() {
            if args.verbose {
                eprintln!("Warning: no data file in {}, skipping", path.display());
            }
            continue;
        }
        if args.verbose {
            println!("Loading {}", ds.display());
        }
        let (feats, labs) = read_dataset(ds.to_str().unwrap())?;
        devices.push(EdgeDevice::new(feats, labs));
    }
    let num_devices = devices.len();
    if num_devices == 0 {
        return Err("No Device_* data found!".into());
    }
    if args.verbose {
        println!("→ Found {} devices\n", num_devices);
    }

    // Track step-specific timing
    let step_start = Instant::now();

    match args.step {
        Step::Setup => {
            //------------------------------------------------------------
            // 3) Local training-update proofs per device
            //------------------------------------------------------------
            if args.verbose {
                println!("--- Client Training Updates ---");
            }
            let mut client_reps = Vec::new();
            let mut total_training_proof_size = 0;
            
            for (i, dev) in devices.iter().enumerate() {
                // Sample SAMPLE_SIZE rows per device (mirrors Python's next_batch)
                let (host_feats, host_labs) = dev.next_batch(SAMPLE_SIZE);
                
                // We can generate multiple proofs from the sampled data
                // For now, let's use the first `bs` samples for the ZK circuit
                if host_feats.len() < args.bs {
                    if args.verbose {
                        eprintln!("Warning: Device {} has fewer samples than ZK batch size", i + 1);
                    }
                    continue;
                }
                
                // DEBUG: Verify we're using the correct batch size
                println!("DEBUG: Device {}, using batch size {} from {} available samples", 
                         i + 1, args.bs, host_feats.len());
                
                // Take the first `bs` samples for the ZK circuit
                let zk_feats: Vec<Vec<Felt>> = host_feats[..args.bs]
                    .iter()
                    .map(|feat_row| feat_row.iter().map(|&v| f64_to_felt(v)).collect())
                    .collect();
                
                let zk_labs: Vec<Vec<Felt>> = host_labs[..args.bs]
                    .iter()
                    .map(|&label| {
                        let (y_onehot, _y_sign) = label_to_one_hot(label, AC, 1e6);
                        y_onehot
                    })
                    .collect();
                
                // Create sign vectors for features (all zeros for now)
                let zk_feats_sign: Vec<Vec<Felt>> = zk_feats
                    .iter()
                    .map(|_| vec![Felt::ZERO; FE])
                    .collect();

                // Fresh random init for each device
                let (init_w, init_w_sign, init_b, init_b_sign) =
                    generate_initial_model(FE, AC, 1.0);
                let lr = f64_to_felt(0.0001);
                let pr = f64_to_felt(1e6);

                // Build & prove (ZK circuit processes all `bs` samples internally)
                let t0 = Instant::now();
                
                // DEBUG: Log what we're passing to the prover
                println!("DEBUG: Creating TrainingUpdateProver with:");
                println!("  - batch_size: {}", args.bs);
                println!("  - x_batch.len(): {}", zk_feats.len());
                println!("  - y_batch.len(): {}", zk_labs.len());
                
                let tp = TrainingUpdateProver::new(
                    proof_options.clone(),
                    init_w.clone(), init_b.clone(),
                    init_w_sign.clone(), init_b_sign.clone(),
                    zk_feats, zk_feats_sign, zk_labs,
                    lr, pr,
                    args.bs, // This is the ZK circuit's batch size
                );
                
                // DEBUG: Build trace and inspect it
                let build_start = Instant::now();
                let trace = tp.build_trace();
                println!("DEBUG: Trace built in {}ms", build_start.elapsed().as_millis());
                println!("DEBUG: Trace dimensions - length: {}, width: {}", 
                         trace.length(), trace.width());
                
                let proof = tp.prove(trace.clone())?;
                let proof_size = proof.to_bytes().len();
                total_training_proof_size += proof_size;
                
                if args.verbose {
                    println!(
                        "Device {:>2}: ZK proof for {} samples: gen = {:>4}ms, size = {} bytes",
                        i + 1,
                        args.bs,
                        t0.elapsed().as_millis(),
                        proof_size
                    );
                    // Add this line specifically for Python script parsing:
                    println!("Training proof size: {} bytes", proof_size);
                }

                // DEBUG: Verify public inputs
                let pub_inputs = tp.get_pub_inputs(&trace);
                println!("DEBUG: Public inputs batch_size: {}", pub_inputs.batch_size);
                println!("DEBUG: Public inputs x_batch.len(): {}", pub_inputs.x_batch.len());
                println!("DEBUG: Public inputs y_batch.len(): {}", pub_inputs.y_batch.len());

                // Verify
                verify::<
                    TrainingUpdateAir,
                    Blake3_256<Felt>,
                    DefaultRandomCoin<Blake3_256<Felt>>,
                    MerkleTree<Blake3_256<Felt>>,
                >(proof, pub_inputs, &AcceptableOptions::OptionSet(vec![proof_options.clone()]))
                .expect("training proof failed!");

                // Extract the result (simplified - you might want to extract final weights)
                client_reps.push(trace.get(0, trace.length() - 1));
            }

            //------------------------------------------------------------
            // 4) Build local_w/local_b from updates
            //------------------------------------------------------------
            let mut local_w = Vec::new();
            let mut local_b = Vec::new();
            for rep in &client_reps {
                let v = rep.as_int() as f64 / 1e6;
                local_w.push(vec![vec![f64_to_felt(v); FE]; AC]);
                local_b.push(vec![f64_to_felt(v); AC]);
            }

            //------------------------------------------------------------
            // 5) Initialize global aggregator
            //------------------------------------------------------------
            let (g_w, _gw_sign, g_b, _gb_sign) =
                generate_initial_model(FE, AC, 10_000.0);
            let k = f64_to_felt(client_reps.len() as f64);

            let agg0 = Instant::now();
            let _agg = GlobalUpdateProver::new(
                proof_options.clone(),
                g_w.clone(), g_b.clone(),
                local_w.clone(), local_b.clone(),
                k,
            );
            if args.verbose {
                println!("Aggregator ready in {}ms\n", agg0.elapsed().as_millis());
                println!("STEP=setup: Generated {} ZK proofs (bs={})", client_reps.len(), args.bs);
                // Output total proof size for Python parsing
                println!("Total training proof size: {} bytes", total_training_proof_size);
            }
        }
        Step::Witness => {
            // Similar debugging for witness step
            let mut client_reps = Vec::new();
            
            for (i, dev) in devices.iter().enumerate() {
                let (host_feats, host_labs) = dev.next_batch(SAMPLE_SIZE);
                
                if host_feats.len() < args.bs {
                    continue;
                }
                
                println!("DEBUG: Witness step - Device {}, batch size {}", i + 1, args.bs);
                
                let zk_feats: Vec<Vec<Felt>> = host_feats[..args.bs]
                    .iter()
                    .map(|feat_row| feat_row.iter().map(|&v| f64_to_felt(v)).collect())
                    .collect();
                
                let zk_labs: Vec<Vec<Felt>> = host_labs[..args.bs]
                    .iter()
                    .map(|&label| {
                        let (y_onehot, _y_sign) = label_to_one_hot(label, AC, 1e6);
                        y_onehot
                    })
                    .collect();
                
                let zk_feats_sign: Vec<Vec<Felt>> = zk_feats
                    .iter()
                    .map(|_| vec![Felt::ZERO; FE])
                    .collect();

                let (init_w, init_w_sign, init_b, init_b_sign) =
                    generate_initial_model(FE, AC, 1.0);
                let lr = f64_to_felt(0.0001);
                let pr = f64_to_felt(1e6);

                let tp = TrainingUpdateProver::new(
                    proof_options.clone(),
                    init_w.clone(), init_b.clone(),
                    init_w_sign.clone(), init_b_sign.clone(),
                    zk_feats, zk_feats_sign, zk_labs,
                    lr, pr,
                    args.bs,
                );
                
                let trace = tp.build_trace();
                println!("DEBUG: Witness trace - length: {}, width: {}", 
                         trace.length(), trace.width());
                
                // For witness step, we just need the trace, not the proof
                client_reps.push(trace.get(0, trace.length() - 1));
            }

            // Build global aggregator
            let mut local_w = Vec::new();
            let mut local_b = Vec::new();
            for rep in &client_reps {
                let v = rep.as_int() as f64 / 1e6;
                local_w.push(vec![vec![f64_to_felt(v); FE]; AC]);
                local_b.push(vec![f64_to_felt(v); AC]);
            }

            let (g_w, _gw_sign, g_b, _gb_sign) =
                generate_initial_model(FE, AC, 10_000.0);
            let k = f64_to_felt(client_reps.len() as f64);

            let agg = GlobalUpdateProver::new(
                proof_options.clone(),
                g_w.clone(), g_b.clone(),
                local_w.clone(), local_b.clone(),
                k,
            );
            
            let t = Instant::now();
            let tr = agg.build_trace();
            if args.verbose {
                println!("witness: {} rows in {}ms", tr.length(), t.elapsed().as_millis());
            }
        }
        Step::Proof => {
            // Similar debugging for proof step
            let mut client_reps = Vec::new();
            let mut total_training_proof_size = 0;
            
            for (i, dev) in devices.iter().enumerate() {
                let (host_feats, host_labs) = dev.next_batch(SAMPLE_SIZE);
                
                if host_feats.len() < args.bs {
                    continue;
                }
                
                println!("DEBUG: Proof step - Device {}, batch size {}", i + 1, args.bs);
                
                let zk_feats: Vec<Vec<Felt>> = host_feats[..args.bs]
                    .iter()
                    .map(|feat_row| feat_row.iter().map(|&v| f64_to_felt(v)).collect())
                    .collect();
                
                let zk_labs: Vec<Vec<Felt>> = host_labs[..args.bs]
                    .iter()
                    .map(|&label| {
                        let (y_onehot, _y_sign) = label_to_one_hot(label, AC, 1e6);
                        y_onehot
                    })
                    .collect();
                
                let zk_feats_sign: Vec<Vec<Felt>> = zk_feats
                    .iter()
                    .map(|_| vec![Felt::ZERO; FE])
                    .collect();

                let (init_w, init_w_sign, init_b, init_b_sign) =
                    generate_initial_model(FE, AC, 1.0);
                let lr = f64_to_felt(0.0001);
                let pr = f64_to_felt(1e6);

                let tp = TrainingUpdateProver::new(
                    proof_options.clone(),
                    init_w.clone(), init_b.clone(),
                    init_w_sign.clone(), init_b_sign.clone(),
                    zk_feats, zk_feats_sign, zk_labs,
                    lr, pr,
                    args.bs,
                );
                
                let trace = tp.build_trace();
                println!("DEBUG: Proof trace - length: {}, width: {}", 
                         trace.length(), trace.width());
                         
                let proof = tp.prove(trace.clone())?;
                let proof_size = proof.to_bytes().len();
                total_training_proof_size += proof_size;
                
                // Verify the training proof
                let pub_inputs = tp.get_pub_inputs(&trace);
                verify::<
                    TrainingUpdateAir,
                    Blake3_256<Felt>,
                    DefaultRandomCoin<Blake3_256<Felt>>,
                    MerkleTree<Blake3_256<Felt>>,
                >(proof, pub_inputs, &AcceptableOptions::OptionSet(vec![proof_options.clone()]))
                .expect("training proof failed!");
                
                client_reps.push(trace.get(0, trace.length() - 1));
            }

            // Build global aggregator
            let mut local_w = Vec::new();
            let mut local_b = Vec::new();
            for rep in &client_reps {
                let v = rep.as_int() as f64 / 1e6;
                local_w.push(vec![vec![f64_to_felt(v); FE]; AC]);
                local_b.push(vec![f64_to_felt(v); AC]);
            }

            let (g_w, _gw_sign, g_b, _gb_sign) =
                generate_initial_model(FE, AC, 10_000.0);
            let k = f64_to_felt(client_reps.len() as f64);

            let agg = GlobalUpdateProver::new(
                proof_options.clone(),
                g_w.clone(), g_b.clone(),
                local_w.clone(), local_b.clone(),
                k,
            );
            
            let t1 = Instant::now();
            let tr = agg.build_trace();
            if args.verbose {
                println!("trace: {} rows in {}ms", tr.length(), t1.elapsed().as_millis());
            }
            
            let t2 = Instant::now();
            let pf = agg.prove(tr.clone())?;
            let aggregation_proof_size = pf.to_bytes().len();
            
            if args.verbose {
                println!("proof: {}ms, {} bytes", t2.elapsed().as_millis(), aggregation_proof_size);
                // Add this line specifically for Python script parsing:
                println!("Proof size: {} bytes", aggregation_proof_size);
                print!("verifying… ");
            }
            
            verify::<
                GlobalUpdateAir,
                Blake3_256<Felt>,
                DefaultRandomCoin<Blake3_256<Felt>>,
                MerkleTree<Blake3_256<Felt>>,
            >(pf, agg.get_pub_inputs(&tr), &AcceptableOptions::OptionSet(vec![proof_options.clone()]))
            .expect("aggregation failed!");
            
            if args.verbose {
                println!("OK");
                // Output summary for Python parsing
                println!("Total training proof size: {} bytes", total_training_proof_size);
                println!("Aggregation proof size: {} bytes", aggregation_proof_size);
                println!("Total proof size: {} bytes", total_training_proof_size + aggregation_proof_size);
            }
        }
    }
    
    if args.verbose {
        println!("\nStep '{}' completed in: {}ms", args.step.to_string().to_lowercase(), step_start.elapsed().as_millis());
        println!("Overall runtime: {}ms", overall_start.elapsed().as_millis());
    }
    Ok(())
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Step::Setup => write!(f, "setup"),
            Step::Witness => write!(f, "witness"),
            Step::Proof => write!(f, "proof"),
        }
    }
}