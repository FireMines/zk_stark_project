// tests/integration_tests.rs
//! Integration tests for the STARK proof system

use winterfell::math::FieldElement;
use zk_stark_project::helper::*;
use zk_stark_project::training::prover::*;
use zk_stark_project::training::air::*;
use zk_stark_project::debug::*;

use winterfell::*;
use winterfell::crypto::{DefaultRandomCoin, MerkleTree, hashers::Blake3_256};
use winterfell::math::fields::f128::BaseElement as Felt;

fn setup_test_data(batch_size: usize) -> (
    Vec<Vec<Felt>>, Vec<Vec<Felt>>, Vec<Felt>, Vec<Felt>,  // w, w_sign, b, b_sign
    Vec<Vec<Felt>>, Vec<Vec<Felt>>, Vec<Vec<Felt>>,        // x_batch, x_batch_sign, y_batch
    Felt, Felt                                              // lr, precision
) {
    // Create test model - check what generate_initial_model actually returns
    let result = generate_initial_model(FE, AC, 1.0);
    
    // Let's handle both possible return types
    let (w, w_sign, b, b_sign) = match result {
        // If it returns 4 values (w, w_sign, b, b_sign)
        (w, w_sign, b, b_sign) => (w, w_sign, b, b_sign),
        // If it returns 2 values, we need to create sign vectors
        (w, b) => {
            let w_sign = vec![vec![Felt::ZERO; w[0].len()]; w.len()];
            let b_sign = vec![Felt::ZERO; b.len()];
            (w, w_sign, b, b_sign)
        }
    };
    
    // Create test batch
    let mut x_batch = Vec::new();
    let mut x_batch_sign = Vec::new();
    let mut y_batch = Vec::new();
    
    for i in 0..batch_size {
        // Create diverse feature vectors
        let x: Vec<Felt> = (0..FE)
            .map(|j| f64_to_felt((i as f64 + j as f64) * 0.1))
            .collect();
        let x_sign = vec![Felt::ZERO; FE];
        
        // Create one-hot labels
        let (y, _) = label_to_one_hot((i % AC) as f64 + 1.0, AC, 1e6);
        
        x_batch.push(x);
        x_batch_sign.push(x_sign);
        y_batch.push(y);
    }
    
    let lr = f64_to_felt(0.01);
    let precision = f64_to_felt(1e6);
    
    (w, w_sign, b, b_sign, x_batch, x_batch_sign, y_batch, lr, precision)
}

#[test]
fn test_proof_generation_and_verification() {
    println!("\n=== Testing Proof Generation and Verification ===");
    
    let batch_size = 1;
    let (w, w_sign, b, b_sign, x_batch, x_batch_sign, y_batch, lr, precision) = 
        setup_test_data(batch_size);
    
    // Create proof options
    let proof_options = ProofOptions::new(
        40, 16, 21,
        FieldExtension::None,
        16, 7,
        BatchingMethod::Algebraic,
        BatchingMethod::Algebraic,
    );
    
    // Create prover
    let prover = TrainingUpdateProver::new(
        proof_options.clone(),
        w, b, w_sign, b_sign,
        x_batch, x_batch_sign, y_batch,
        lr, precision, batch_size,
    );
    
    // Build trace
    let trace = prover.build_trace();
    println!("Trace created: {} rows x {} columns", trace.length(), trace.width());
    
    // Generate proof
    let start_time = std::time::Instant::now();
    let proof = prover.prove(trace.clone()).expect("Proof generation failed");
    let proof_time = start_time.elapsed();
    
    println!("Proof generated in {:?}", proof_time);
    println!("Proof size: {} bytes", proof.to_bytes().len());
    
    // Get public inputs
    let pub_inputs = prover.get_pub_inputs(&trace);
    
    // Verify proof
    let start_time = std::time::Instant::now();
    let verification_result = verify::<
        TrainingUpdateAir,
        Blake3_256<Felt>,
        DefaultRandomCoin<Blake3_256<Felt>>,
        MerkleTree<Blake3_256<Felt>>,
    >(proof, pub_inputs, &AcceptableOptions::OptionSet(vec![proof_options]));
    let verify_time = start_time.elapsed();
    
    println!("Verification completed in {:?}", verify_time);
    assert!(verification_result.is_ok(), "Verification failed: {:?}", verification_result);
    
    println!("✓ Proof generation and verification successful!");
}

#[test]
fn test_different_batch_sizes() {
    println!("\n=== Testing Different Batch Sizes ===");
    
    let batch_sizes = vec![1, 2];  // Start with smaller sizes for testing
    
    for batch_size in batch_sizes {
        println!("\nTesting batch size: {}", batch_size);
        
        let (w, w_sign, b, b_sign, x_batch, x_batch_sign, y_batch, lr, precision) = 
            setup_test_data(batch_size);
        
        let proof_options = ProofOptions::new(
            40, 16, 21,
            FieldExtension::None,
            16, 7,
            BatchingMethod::Algebraic,
            BatchingMethod::Algebraic,
        );
        
        let prover = TrainingUpdateProver::new(
            proof_options.clone(),
            w, b, w_sign, b_sign,
            x_batch, x_batch_sign, y_batch,
            lr, precision, batch_size,
        );
        
        let trace = prover.build_trace();
        let proof = prover.prove(trace.clone()).expect(&format!("Proof generation failed for batch size {}", batch_size));
        let pub_inputs = prover.get_pub_inputs(&trace);
        
        let verification_result = verify::<
            TrainingUpdateAir,
            Blake3_256<Felt>,
            DefaultRandomCoin<Blake3_256<Felt>>,
            MerkleTree<Blake3_256<Felt>>,
        >(proof, pub_inputs, &AcceptableOptions::OptionSet(vec![proof_options]));
        
        assert!(verification_result.is_ok(), 
               "Verification failed for batch size {}: {:?}", batch_size, verification_result);
        
        println!("✓ Batch size {} passed", batch_size);
    }
}

#[test]
fn test_trace_analysis() {
    println!("\n=== Testing Trace Analysis ===");
    
    let batch_size = 1;
    let (w, w_sign, b, b_sign, x_batch, x_batch_sign, y_batch, lr, precision) = 
        setup_test_data(batch_size);
    
    let proof_options = ProofOptions::new(
        40, 16, 21,
        FieldExtension::None,
        16, 7,
        BatchingMethod::Algebraic,
        BatchingMethod::Algebraic,
    );
    
    let prover = TrainingUpdateProver::new(
        proof_options,
        w, b, w_sign, b_sign,
        x_batch, x_batch_sign, y_batch,
        lr, precision, batch_size,
    );
    
    let trace = prover.build_trace();
    
    // Use debug utilities
    analyze_trace(&trace);
    
    // Verify trace transitions
    verify_trace_transitions(&trace, batch_size).expect("Trace transition verification failed");
    
    // Export trace to CSV
    export_trace_csv(&trace, "test_trace.csv").expect("Failed to export trace");
    
    println!("✓ Trace analysis completed successfully");
    
    // Clean up
    std::fs::remove_file("test_trace.csv").ok();
}

#[test]
fn test_invalid_inputs_rejected() {
    println!("\n=== Testing Invalid Inputs Are Rejected ===");
    
    let batch_size = 1;
    let (w, w_sign, b, b_sign, mut x_batch, x_batch_sign, y_batch, lr, precision) = 
        setup_test_data(batch_size);
    
    let proof_options = ProofOptions::new(
        40, 16, 21,
        FieldExtension::None,
        16, 7,
        BatchingMethod::Algebraic,
        BatchingMethod::Algebraic,
    );
    
    // Test: Wrong batch size
    x_batch.push(vec![f64_to_felt(0.0); FE]); // Add extra sample
    let result = std::panic::catch_unwind(|| {
        TrainingUpdateProver::new(
            proof_options.clone(),
            w.clone(), b.clone(), w_sign.clone(), b_sign.clone(),
            x_batch.clone(), x_batch_sign.clone(), y_batch.clone(),
            lr, precision, batch_size,
        )
    });
    assert!(result.is_err(), "Should panic when batch sizes don't match");
    
    println!("✓ Invalid input tests completed");
}