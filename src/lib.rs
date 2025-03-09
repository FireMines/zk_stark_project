// lib.rs

use std::ffi::{CString};
use std::os::raw::c_char;

use winterfell::{
    Air, AirContext, Assertion, ByteWriter, EvaluationFrame, FieldExtension,
    ProofOptions, Prover, Trace, TraceInfo, TraceTable,
    TransitionConstraintDegree,
    crypto::{DefaultRandomCoin, MerkleTree, hashers::Blake3_256},
    DefaultTraceLde,
    DefaultConstraintEvaluator,
    DefaultConstraintCommitment,
    matrix::ColMatrix,
    BatchingMethod,
    StarkDomain,
    TracePolyTable,
    PartitionOptions,
    CompositionPolyTrace,
    CompositionPoly,
    ConstraintCompositionCoefficients,
    AuxRandElements,
};
use winterfell::math::{FieldElement, ToElements};
use winterfell::math::fields::f128::BaseElement as Felt;
use winter_utils::Serializable;

// -------------------------------------------
// Reuse the training and aggregation modules
// (see your original code for details)
// -------------------------------------------
mod training {
    use super::*;
    #[derive(Clone)]
    pub struct TrainingInputs {
        pub initial: Felt,
        pub final_weight: Felt,
        pub steps: usize,
    }
    
    impl Serializable for TrainingInputs {
        fn write_into<W: ByteWriter>(&self, target: &mut W) {
            target.write(self.initial);
            target.write(self.final_weight);
            target.write(Felt::new(self.steps as u128));
        }
    }
    
    impl ToElements<Felt> for TrainingInputs {
        fn to_elements(&self) -> Vec<Felt> {
            vec![self.initial, self.final_weight, Felt::new(self.steps as u128)]
        }
    }
    
    pub struct TrainingAir {
        context: AirContext<Felt>,
        initial: Felt,
        final_weight: Felt,
    }
    
    impl Air for TrainingAir {
        type BaseField = Felt;
        type PublicInputs = TrainingInputs;
    
        fn new(trace_info: TraceInfo, pub_inputs: TrainingInputs, options: ProofOptions) -> Self {
            let degrees = vec![TransitionConstraintDegree::new(1)];
            let context = AirContext::new(trace_info, degrees, 2, options);
            Self {
                context,
                initial: pub_inputs.initial,
                final_weight: pub_inputs.final_weight,
            }
        }
    
        fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            frame: &EvaluationFrame<E>,
            _periodic_values: &[E],
            result: &mut [E],
        ) {
            debug_assert_eq!(frame.current().len(), 2);
            debug_assert_eq!(frame.next().len(), 2);
    
            let weight_current = frame.current()[0];
            let gradient = frame.current()[1];
            let weight_next = frame.next()[0];
            result[0] = weight_current - gradient - weight_next;
        }
    
        fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
            let trace_len = self.trace_length();
            vec![
                Assertion::single(0, 0, self.initial),
                Assertion::single(0, trace_len - 1, self.final_weight),
            ]
        }
    
        fn context(&self) -> &AirContext<Self::BaseField> {
            &self.context
        }
    }
    
    pub struct TrainingProver {
        pub options: ProofOptions,
        pub gradients: Vec<Felt>,
        pub trace_length: usize,
        pub initial: Felt,
    }
    
    impl TrainingProver {
        pub fn new(options: ProofOptions, initial: Felt, gradients: Vec<Felt>) -> Self {
            let real_length = gradients.len() + 1;
            let mut trace_length = real_length.next_power_of_two();
            if trace_length < 8 {
                trace_length = 8;
            }
            Self { options, gradients, trace_length, initial }
        }
    
        pub fn build_trace(&self) -> TraceTable<Felt> {
            let mut weight_trace = Vec::with_capacity(self.trace_length);
            let mut gradient_trace = Vec::with_capacity(self.trace_length);
            weight_trace.push(self.initial);
            for &g in &self.gradients {
                let prev = *weight_trace.last().unwrap();
                let next = prev - g;
                weight_trace.push(next);
                gradient_trace.push(g);
            }
            while weight_trace.len() < self.trace_length {
                let prev = *weight_trace.last().unwrap();
                weight_trace.push(prev);
                gradient_trace.push(Felt::new(0));
            }
            if gradient_trace.len() < weight_trace.len() {
                gradient_trace.push(Felt::new(0));
            }
            TraceTable::init(vec![weight_trace, gradient_trace])
        }
    }
    
    impl Prover for TrainingProver {
        type BaseField = Felt;
        type Air = TrainingAir;
        type Trace = TraceTable<Felt>;
        type HashFn = Blake3_256<Felt>;
        type VC = MerkleTree<Self::HashFn>;
        type RandomCoin = DefaultRandomCoin<Self::HashFn>;
        type TraceLde<E: FieldElement<BaseField = Self::BaseField>> = DefaultTraceLde<E, Self::HashFn, Self::VC>;
        type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
            DefaultConstraintEvaluator<'a, Self::Air, E>;
        type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
            DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;
    
        fn get_pub_inputs(&self, trace: &Self::Trace) -> TrainingInputs {
            TrainingInputs {
                initial: trace.get(0, 0),
                final_weight: trace.get(0, trace.length() - 1),
                steps: self.gradients.len(),
            }
        }
    
        fn options(&self) -> &ProofOptions {
            &self.options
        }
    
        fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            trace_info: &TraceInfo,
            main_trace: &ColMatrix<Self::BaseField>,
            domain: &StarkDomain<Self::BaseField>,
            partition_options: PartitionOptions,
        ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
            DefaultTraceLde::new(trace_info, main_trace, domain, partition_options)
        }
    
        fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            air: &'a Self::Air,
            aux_rand_elements: Option<AuxRandElements<E>>,
            composition_coefficients: ConstraintCompositionCoefficients<E>,
        ) -> Self::ConstraintEvaluator<'a, E> {
            DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
        }
    
        fn build_constraint_commitment<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            composition_poly_trace: CompositionPolyTrace<E>,
            num_constraint_composition_columns: usize,
            domain: &StarkDomain<Self::BaseField>,
            partition_options: PartitionOptions,
        ) -> (Self::ConstraintCommitment<E>, CompositionPoly<E>) {
            DefaultConstraintCommitment::new(
                composition_poly_trace,
                num_constraint_composition_columns,
                domain,
                partition_options,
            )
        }
    }
}

mod aggregation {
    use super::*;
    
    #[derive(Clone)]
    pub struct AggregationInputs {
        pub initial: Felt,
        pub aggregated: Felt,
        pub count: usize,
    }
    
    impl Serializable for AggregationInputs {
        fn write_into<W: ByteWriter>(&self, target: &mut W) {
            target.write(self.initial);
            target.write(self.aggregated);
            target.write(Felt::new(self.count as u128));
        }
    }
    
    impl ToElements<Felt> for AggregationInputs {
        fn to_elements(&self) -> Vec<Felt> {
            vec![self.initial, self.aggregated, Felt::new(self.count as u128)]
        }
    }
    
    pub struct AggregationAir {
        context: AirContext<Felt>,
        initial: Felt,
        aggregated: Felt,
    }
    
    impl Air for AggregationAir {
        type BaseField = Felt;
        type PublicInputs = AggregationInputs;
    
        fn new(trace_info: TraceInfo, pub_inputs: AggregationInputs, options: ProofOptions) -> Self {
            let degrees = vec![TransitionConstraintDegree::new(1)];
            let context = AirContext::new(trace_info, degrees, 2, options);
            Self {
                context,
                initial: pub_inputs.initial,
                aggregated: pub_inputs.aggregated,
            }
        }
    
        fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            frame: &EvaluationFrame<E>,
            _periodic_values: &[E],
            result: &mut [E],
        ) {
            debug_assert_eq!(frame.current().len(), 2);
            debug_assert_eq!(frame.next().len(), 2);
    
            let sum_current = frame.current()[0];
            let update = frame.current()[1];
            let sum_next = frame.next()[0];
            result[0] = sum_current + update - sum_next;
        }
    
        fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
            let trace_len = self.trace_length();
            vec![
                Assertion::single(0, 0, self.initial),
                Assertion::single(0, trace_len - 1, self.aggregated),
            ]
        }
    
        fn context(&self) -> &AirContext<Self::BaseField> {
            &self.context
        }
    }
    
    pub struct AggregationProver {
        pub options: ProofOptions,
        pub updates: Vec<Felt>,
        pub trace_length: usize,
    }
    
    impl AggregationProver {
        pub fn new(options: ProofOptions, updates: Vec<Felt>) -> Self {
            let real_length = updates.len() + 1;
            let mut trace_length = real_length.next_power_of_two();
            if trace_length < 8 {
                trace_length = 8;
            }
            Self { options, updates, trace_length }
        }
    
        pub fn build_trace(initial: Felt, updates: &[Felt], trace_length: usize) -> TraceTable<Felt> {
            let mut sum_trace = Vec::with_capacity(trace_length);
            let mut update_trace = Vec::with_capacity(trace_length);
            sum_trace.push(initial);
            for &u in updates {
                let prev = *sum_trace.last().unwrap();
                let next = prev + u;
                sum_trace.push(next);
                update_trace.push(u);
            }
            while sum_trace.len() < trace_length {
                let prev = *sum_trace.last().unwrap();
                sum_trace.push(prev);
                update_trace.push(Felt::new(0));
            }
            if update_trace.len() < sum_trace.len() {
                update_trace.push(Felt::new(0));
            }
            TraceTable::init(vec![sum_trace, update_trace])
        }
    }
    
    impl Prover for AggregationProver {
        type BaseField = Felt;
        type Air = AggregationAir;
        type Trace = TraceTable<Felt>;
        type HashFn = Blake3_256<Felt>;
        type VC = MerkleTree<Self::HashFn>;
        type RandomCoin = DefaultRandomCoin<Self::HashFn>;
        type TraceLde<E: FieldElement<BaseField = Self::BaseField>> = DefaultTraceLde<E, Self::HashFn, Self::VC>;
        type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
            DefaultConstraintEvaluator<'a, Self::Air, E>;
        type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
            DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;
    
        fn get_pub_inputs(&self, trace: &Self::Trace) -> AggregationInputs {
            let last = trace.length() - 1;
            AggregationInputs {
                initial: trace.get(0, 0),
                aggregated: trace.get(0, last),
                count: self.updates.len(),
            }
        }
    
        fn options(&self) -> &ProofOptions {
            &self.options
        }
    
        fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            trace_info: &TraceInfo,
            main_trace: &ColMatrix<Self::BaseField>,
            domain: &StarkDomain<Self::BaseField>,
            partition_options: PartitionOptions,
        ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
            DefaultTraceLde::new(trace_info, main_trace, domain, partition_options)
        }
    
        fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            air: &'a Self::Air,
            aux_rand_elements: Option<AuxRandElements<E>>,
            composition_coefficients: ConstraintCompositionCoefficients<E>,
        ) -> Self::ConstraintEvaluator<'a, E> {
            DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
        }
    
        fn build_constraint_commitment<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            composition_poly_trace: CompositionPolyTrace<E>,
            num_constraint_composition_columns: usize,
            domain: &StarkDomain<Self::BaseField>,
            partition_options: PartitionOptions,
        ) -> (Self::ConstraintCommitment<E>, CompositionPoly<E>) {
            DefaultConstraintCommitment::new(
                composition_poly_trace,
                num_constraint_composition_columns,
                domain,
                partition_options,
            )
        }
    }
}

// ----------------------------------------------------
// FFI Interface: Expose functions to generate proofs.
// These functions return a JSON string (allocated on the heap)
// containing hex-encoded "proof" and "pub_inputs".
// ----------------------------------------------------

#[no_mangle]
pub extern "C" fn generate_training_proof(
    initial: u128,
    gradients_ptr: *const u128,
    gradients_len: usize,
) -> *mut c_char {
    // Safety: create a slice from the pointer.
    if gradients_ptr.is_null() {
        let err = CString::new("{\"error\": \"Null pointer for gradients\"}").unwrap();
        return err.into_raw();
    }
    let gradients_slice = unsafe { std::slice::from_raw_parts(gradients_ptr, gradients_len) };
    let gradients: Vec<Felt> = gradients_slice.iter().map(|&x| Felt::new(x)).collect();

    // Set up proof options (adjust as needed)
    let proof_options = ProofOptions::new(
        40,                       // security level
        16,                       // blowup factor
        21,                       // grinding factor
        FieldExtension::None,     // field extension
        16,                       // fri_max_degree parameter
        7,                        // fri_layout parameter
        BatchingMethod::Algebraic,
        BatchingMethod::Algebraic,
    );
    
    let training_prover = training::TrainingProver::new(proof_options, Felt::new(initial), gradients);
    let trace = training_prover.build_trace();
    let pub_inputs = training_prover.get_pub_inputs(&trace);
    
    let proof = match training_prover.prove(trace) {
        Ok(p) => p,
        Err(e) => {
            let err_str = format!("{{\"error\": \"Proof generation failed: {}\"}}", e);
            let err_cstring = CString::new(err_str).unwrap();
            return err_cstring.into_raw();
        }
    };
    
    let proof_bytes = proof.to_bytes();
    let proof_hex = hex::encode(proof_bytes);
    
    let mut pub_inputs_bytes = Vec::new();
    pub_inputs.write_into(&mut pub_inputs_bytes);
    let pub_inputs_hex = hex::encode(pub_inputs_bytes);
    
    let result_json = format!(
        "{{\"proof\": \"{}\", \"pub_inputs\": \"{}\"}}",
        proof_hex, pub_inputs_hex
    );
    let c_string = CString::new(result_json).unwrap();
    c_string.into_raw()
}

#[no_mangle]
pub extern "C" fn generate_aggregation_proof(
    initial: u128,
    updates_ptr: *const u128,
    updates_len: usize,
) -> *mut c_char {
    if updates_ptr.is_null() {
        let err = CString::new("{\"error\": \"Null pointer for updates\"}").unwrap();
        return err.into_raw();
    }
    let updates_slice = unsafe { std::slice::from_raw_parts(updates_ptr, updates_len) };
    let updates: Vec<Felt> = updates_slice.iter().map(|&x| Felt::new(x)).collect();
    
    let proof_options = ProofOptions::new(
        40,
        16,
        21,
        FieldExtension::None,
        16,
        7,
        BatchingMethod::Algebraic,
        BatchingMethod::Algebraic,
    );
    
    let aggregation_prover = aggregation::AggregationProver::new(proof_options, updates);
    let trace = aggregation::AggregationProver::build_trace(Felt::new(initial), &aggregation_prover.updates, aggregation_prover.trace_length);
    let pub_inputs = aggregation_prover.get_pub_inputs(&trace);
    
    let proof = match aggregation_prover.prove(trace) {
        Ok(p) => p,
        Err(e) => {
            let err_str = format!("{{\"error\": \"Proof generation failed: {}\"}}", e);
            let err_cstring = CString::new(err_str).unwrap();
            return err_cstring.into_raw();
        }
    };
    
    let proof_bytes = proof.to_bytes();
    let proof_hex = hex::encode(proof_bytes);
    
    let mut pub_inputs_bytes = Vec::new();
    pub_inputs.write_into(&mut pub_inputs_bytes);
    let pub_inputs_hex = hex::encode(pub_inputs_bytes);
    
    let result_json = format!(
        "{{\"proof\": \"{}\", \"pub_inputs\": \"{}\"}}",
        proof_hex, pub_inputs_hex
    );
    let c_string = CString::new(result_json).unwrap();
    c_string.into_raw()
}

/// Call this function from Python to free the memory allocated for the returned string.
#[no_mangle]
pub extern "C" fn free_rust_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        CString::from_raw(s);
    }
}
