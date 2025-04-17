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
    read_dataset, split_dataset,
    f64_to_felt, generate_initial_model, label_to_one_hot,
    AC, C, FE,
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
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::from_args();
    let overall_start = Instant::now();

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
    let (features, labels) = read_dataset("devices/edge_device/data/train.txt")?;
    let client_data = split_dataset(features, labels, args.clients);

    //------------------------------------------------------------
    // 3) Run the *one‑step* training update for each client
    //------------------------------------------------------------
    println!("--- Client Training Updates ---");
    let mut client_reps = Vec::with_capacity(args.clients);
    for (i, (client_feats, client_labs)) in client_data.iter().enumerate() {
        // we'll train on the *first* sample of each shard
        let x_vals: Vec<Felt> = client_feats[0].iter().map(|&v| f64_to_felt(v)).collect();
        let x_signs     = vec![Felt::ZERO; FE];
        let (y_onehot, _y_sign) = 
            label_to_one_hot(client_labs[0], AC, /*precision=*/1e6);

        // fresh random init per client
        let (init_w, init_w_sign, init_b, init_b_sign) =
            generate_initial_model(FE, AC, /*σ=*/1.0);
        let lr = f64_to_felt(0.0001);
        let pr = f64_to_felt(1e6);

        // build & prove training‑update STARK
        let t0 = Instant::now();
        let train_prover = TrainingUpdateProver::new(
            proof_options.clone(),
            init_w.clone(),  init_b.clone(),
            init_w_sign.clone(), init_b_sign.clone(),
            x_vals.clone(),  x_signs.clone(),
            y_onehot.clone(),
            lr, pr,
        );
        let train_trace = train_prover.build_trace();
        let train_proof = train_prover.prove(train_trace.clone())?;
        let dt = t0.elapsed();
        println!(
            "Client {:>2}: training proof gen = {:>4} ms, size = {} bytes",
            i+1,
            dt.as_millis(),
            train_proof.to_bytes().len(),
        );

        // verify training proof
        let train_pub = train_prover.get_pub_inputs(&train_trace);
        verify::<
            TrainingUpdateAir,
            Blake3_256<Felt>,
            DefaultRandomCoin<Blake3_256<Felt>>,
            MerkleTree<Blake3_256<Felt>>
        >(train_proof, train_pub.clone(), &AcceptableOptions::OptionSet(vec![proof_options.clone()]))
            .expect("training proof failed!");
        println!("  ↳ client {:>2} verified OK\n", i+1);

        // collect the single‐coordinate update
        client_reps.push(train_pub.final_state[0]);
    }

    //------------------------------------------------------------
    // 4) Build local_w, local_b from those scalars
    //------------------------------------------------------------
    let mut local_w = Vec::with_capacity(args.clients);
    let mut local_b = Vec::with_capacity(args.clients);
    for rep in &client_reps {
        let val = rep.as_int() as f64 / 1e6;
        let w_mat = vec![vec![f64_to_felt(val); FE]; AC];
        let b_vec = vec![f64_to_felt(val); AC];
        local_w.push(w_mat);
        local_b.push(b_vec);
    }

    //------------------------------------------------------------
    // 5) Initialize global model
    //------------------------------------------------------------
    let (raw_global_w, _gw_sign, raw_global_b, _gb_sign) =
        generate_initial_model(FE, AC, /*σ=*/10_000.0);
    let k = f64_to_felt(args.clients as f64);

    //------------------------------------------------------------
    // 6) Create aggregator prover once
    //------------------------------------------------------------
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
        "Aggregator prover instantiated in {} ms\n",
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
                "  ↳ built trace with {} rows in {} ms",
                trace.length(),
                t0.elapsed().as_millis()
            );
        }

        Step::Proof => {
            println!("STEP=proof: build trace, generate + verify aggregator proof…");

            let t1 = Instant::now();
            let trace = aggregator_prover.build_trace();
            println!(
                "  ↳ trace with {} rows in {} ms",
                trace.length(),
                t1.elapsed().as_millis()
            );

            let t2 = Instant::now();
            let proof = aggregator_prover.prove(trace.clone())?;
            println!(
                "  ↳ proof generated in {} ms, size = {} bytes",
                t2.elapsed().as_millis(),
                proof.to_bytes().len()
            );

            print!("  ↳ verifying… ");
            let pub_inputs = aggregator_prover.get_pub_inputs(&trace);
            verify::<
                GlobalUpdateAir,
                Blake3_256<Felt>,
                DefaultRandomCoin<Blake3_256<Felt>>,
                MerkleTree<Blake3_256<Felt>>
            >(proof, pub_inputs, &AcceptableOptions::OptionSet(vec![proof_options.clone()]))
                .expect("aggregation proof failed!");
            println!("OK");
        }
    }

    println!(
        "\nOverall run time: {} ms",
        overall_start.elapsed().as_millis()
    );
    Ok(())
}
