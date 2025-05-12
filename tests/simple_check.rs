// tests/simple_check.rs
//! Simple test to check what functions are available

use zk_stark_project::helper::*;
use winterfell::math::{fields::f128::BaseElement as Felt, FieldElement};

#[test]
fn test_generate_initial_model_signature() {
    // Let's check what generate_initial_model actually returns
    let result = generate_initial_model(FE, AC, 1.0);
    
    // This will help us understand the return type
    println!("generate_initial_model returned: {:?}", &result);
    
    // If it returns 2 elements, we can adapt our tests
    let (w, b) = result;
    println!("w shape: {}x{}", w.len(), w[0].len());
    println!("b length: {}", b.len());
    
    // We can create sign vectors manually if needed
    let w_sign = vec![vec![Felt::ZERO; w[0].len()]; w.len()];
    let b_sign = vec![Felt::ZERO; b.len()];
    
    assert_eq!(w.len(), AC);
    assert_eq!(w[0].len(), FE);
    assert_eq!(b.len(), AC);
    assert_eq!(w_sign.len(), AC);
    assert_eq!(b_sign.len(), AC);
}

#[test]
fn test_basic_functionality() {
    // Test basic field operations
    let a = f64_to_felt(3.0);
    let b = f64_to_felt(2.0);
    
    // Test addition
    let (sum, sign) = add(a, b, Felt::ZERO, Felt::ZERO);
    assert_eq!(sign, Felt::ZERO);
    
    // Test that the result is reasonable
    let sum_f64 = sum.as_int() as f64 / 1e6;
    assert!((sum_f64 - 5.0).abs() < 0.001);
}

#[test]
fn test_one_hot_encoding() {
    let (y_onehot, _y_sign) = label_to_one_hot(1.0, AC, 1e6);
    assert_eq!(y_onehot.len(), AC);
    
    // Check that exactly one element is non-zero
    let non_zero_count = y_onehot.iter().filter(|&&x| x != Felt::ZERO).count();
    assert_eq!(non_zero_count, 1);
}

#[test]
fn test_constants() {
    println!("AC = {}", AC);
    println!("FE = {}", FE);
    
    // These should be positive
    assert!(AC > 0);
    assert!(FE > 0);
}