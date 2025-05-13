// src/training/air.rs

use crate::helper::{bit_constraint, f64_to_felt, EvaluationFrameExt};
use crate::signed::{add_generic as add, sub_generic as sub,
    mul_generic as mul, div_generic as div};

use winterfell::{
    Air, AirContext, Assertion, ByteWriter, EvaluationFrame, ProofOptions,
    TraceInfo, TransitionConstraintDegree,
};
use winter_utils::Serializable;
use winterfell::math::{FieldElement, ToElements};
use winterfell::math::fields::f128::BaseElement as Felt;

/// Public inputs for the masked zk‑STARK of one SGD step with batch processing.
/// We only assert masked state boundaries, so raw values remain hidden.
#[derive(Clone)]
pub struct TrainingUpdateInputs {
    /// masked initial flattened state [v0+s0, v1+s1, …]
    pub initial_masked: Vec<Felt>,
    /// masked final flattened state
    pub final_masked: Vec<Felt>,
    /// number of steps = trace_length − 1
    pub steps: usize,
    /// public batch of features and labels
    pub x_batch: Vec<Vec<Felt>>,    // BS × FE
    pub y_batch: Vec<Vec<Felt>>,    // BS × AC  
    /// public scalar params
    pub learning_rate: Felt,
    pub precision: Felt,
    /// batch size
    pub batch_size: usize,
}

impl Serializable for TrainingUpdateInputs {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        // DEBUG: Log serialization
        println!("DEBUG: TrainingUpdateInputs::write_into called");
        println!("  - batch_size: {}", self.batch_size);
        println!("  - x_batch.len(): {}", self.x_batch.len());
        println!("  - y_batch.len(): {}", self.y_batch.len());
        
        // write masked initial, masked final, then steps
        for &v in &self.initial_masked {
            target.write(v);
        }
        for &v in &self.final_masked {
            target.write(v);
        }
        target.write(f64_to_felt(self.steps as f64));
        
        // write batch data
        target.write(f64_to_felt(self.batch_size as f64));
        for batch_x in &self.x_batch {
            for &v in batch_x {
                target.write(v);
            }
        }
        for batch_y in &self.y_batch {
            for &v in batch_y {
                target.write(v);
            }
        }
        
        target.write(self.learning_rate);
        target.write(self.precision);
    }
}

impl ToElements<Felt> for TrainingUpdateInputs {
    fn to_elements(&self) -> Vec<Felt> {
        // DEBUG: Log to_elements
        println!("DEBUG: TrainingUpdateInputs::to_elements called");
        
        let mut v = self.initial_masked.clone();
        v.extend(self.final_masked.iter());
        v.push(f64_to_felt(self.steps as f64));
        v.push(f64_to_felt(self.batch_size as f64));
        
        // flatten batch data
        for batch_x in &self.x_batch {
            v.extend(batch_x.iter());
        }
        for batch_y in &self.y_batch {
            v.extend(batch_y.iter());
        }
        
        v.push(self.learning_rate);
        v.push(self.precision);
        
        println!("  - total elements: {}", v.len());
        v
    }
}

pub struct TrainingUpdateAir {
    ctx: AirContext<Felt>,
    pub_inputs: TrainingUpdateInputs,
}

impl Air for TrainingUpdateAir {
    type BaseField = Felt;
    type PublicInputs = TrainingUpdateInputs;

    fn new(ti: TraceInfo, pub_inputs: TrainingUpdateInputs, opt: ProofOptions) -> Self {
        let width = ti.width();
        let degrees = vec![TransitionConstraintDegree::new(1); width];
        
        // DEBUG: Log AIR creation
        println!("DEBUG: TrainingUpdateAir::new called");
        println!("  - trace width: {}", width);
        println!("  - trace length: {}", ti.length());
        println!("  - pub_inputs.batch_size: {}", pub_inputs.batch_size);
        println!("  - pub_inputs.steps: {}", pub_inputs.steps);
        println!("  - pub_inputs.x_batch.len(): {}", pub_inputs.x_batch.len());
        println!("  - pub_inputs.y_batch.len(): {}", pub_inputs.y_batch.len());
        
        // Validate that batch size matches the data
        assert_eq!(pub_inputs.x_batch.len(), pub_inputs.batch_size,
                   "x_batch size doesn't match batch_size in public inputs");
        assert_eq!(pub_inputs.y_batch.len(), pub_inputs.batch_size,
                   "y_batch size doesn't match batch_size in public inputs");
        
        Self { 
            ctx: AirContext::new(ti, degrees, width, opt), 
            pub_inputs,
        }
    }

    fn get_assertions(&self) -> Vec<Assertion<Felt>> {
        let width = self.ctx.trace_info().width() / 2; // half is masked state
        let n = self.ctx.trace_len() - 1;
        let mut assertions = Vec::with_capacity(width * 2);
        
        // DEBUG: Log assertions
        println!("DEBUG: TrainingUpdateAir::get_assertions called");
        println!("  - width (half trace): {}", width);
        println!("  - final step (n): {}", n);
        
        // assert masked initial at row 0
        for i in 0..width {
            assertions.push(Assertion::single(i, 0, self.pub_inputs.initial_masked[i]));
        }
        // assert masked final at row n
        for i in 0..width {
            assertions.push(Assertion::single(i, n, self.pub_inputs.final_masked[i]));
        }
        
        println!("  - total assertions: {}", assertions.len());
        assertions
    }

    #[allow(clippy::too_many_lines)]
    fn evaluate_transition<E: FieldElement<BaseField = Felt>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic: &[E],
        result: &mut [E],
    ) {
        let width = self.ctx.trace_info().width();
        let m = width / 2;
        
        // Get current step (0-indexed)
        let current_step = frame.current_step();
        
        // DEBUG: Log constraint evaluation for first few steps
        if current_step <= 5 || (current_step <= self.pub_inputs.batch_size && current_step % 10 == 0) {
            println!("DEBUG: evaluate_transition for step {} (batch_size: {})", 
                     current_step, self.pub_inputs.batch_size);
        }
        
        // Only apply training constraints for steps within batch size
        if current_step > 0 && current_step <= self.pub_inputs.batch_size {
            let sample_idx = current_step - 1;
            
            // DEBUG: Log constraint application
            if sample_idx < 5 {
                println!("DEBUG: Applying constraints for step {} (sample {})", 
                         current_step, sample_idx);
            }
            
            // Validate sample index
            if sample_idx >= self.pub_inputs.x_batch.len() {
                panic!("Sample index {} out of bounds for x_batch (len: {})", 
                       sample_idx, self.pub_inputs.x_batch.len());
            }
            
            // Get dimensions for this sample
            let fe = self.pub_inputs.x_batch[sample_idx].len();
            let ac = self.pub_inputs.y_batch[sample_idx].len();
            
            let two = E::from(f64_to_felt(2.0));
            let pr  = E::from(self.pub_inputs.precision);
            let lr  = E::from(self.pub_inputs.learning_rate);
            let zero = E::ZERO;

            let cur = frame.current();
            let nxt = frame.next();

            // helpers to access masked vs mask columns
            let masked_cur = |i| cur[i];
            let mask_cur   = |i| cur[m + i];
            let masked_nxt = |i| nxt[i];
            let mask_nxt   = |i| nxt[m + i];

            // forward pass: compute raw errors for this sample
            let mut err = vec![zero; ac];
            let mut serr = vec![zero; ac];
            for j in 0..ac {
                let idx_w  = |i| 2 * (j * fe + i);
                let idx_ws = |i| idx_w(i) + 1;
                let idx_b  = 2 * (ac * fe + j);
                let idx_bs = idx_b + 1;

                let mut dot = zero;
                let mut sdot = zero;
                for i in 0..fe {
                    let raw_v = masked_cur(idx_w(i)) - mask_cur(idx_w(i));
                    let raw_s = masked_cur(idx_ws(i)) - mask_cur(idx_ws(i));
                    let (p, sp) = mul(raw_v, raw_s, E::from(self.pub_inputs.x_batch[sample_idx][i]), zero);
                    let (nd, snd) = add(dot, sdot, p, sp);
                    dot = nd; sdot = snd;
                }
                let (q, sdiv) = div(dot, sdot, pr, zero);

                let raw_b  = masked_cur(idx_b) - mask_cur(idx_b);
                let raw_bs = masked_cur(idx_bs) - mask_cur(idx_bs);
                let (pred, spred) = add(q, sdiv, raw_b, raw_bs);

                let (basic, sbasic) = sub(pred, spred, E::from(self.pub_inputs.y_batch[sample_idx][j]), zero);
                let (num, snum)     = mul(basic, sbasic, two, zero);
                let (scaled, sscaled) =
                    div(num, snum, E::from(f64_to_felt(ac as f64)), zero);
                err[j] = scaled; serr[j] = sscaled;
            }

            // backward pass + constraints for this sample
            for j in 0..ac {
                for i in 0..fe {
                    let idx_v = 2 * (j * fe + i);
                    let idx_s = idx_v + 1;

                    let (p, sp) = mul(err[j], serr[j], E::from(self.pub_inputs.x_batch[sample_idx][i]), zero);
                    let (t, st) = div(p, sp, lr, zero);
                    let (grad, sg) = div(t, st, pr, zero);

                    let raw_v = masked_cur(idx_v) - mask_cur(idx_v);
                    let raw_s = masked_cur(idx_s) - mask_cur(idx_s);
                    let (exp, sexp) = sub(raw_v, raw_s, grad, sg);

                    result[idx_v] =
                        masked_nxt(idx_v) - (exp + mask_nxt(idx_v));
                    result[idx_s] =
                        masked_nxt(idx_s) - (sexp + mask_nxt(idx_s));
                    result[idx_s] += bit_constraint(raw_s);
                }
                let idx_b  = 2 * (ac * fe + j);
                let idx_bs = idx_b + 1;

                let (t, st) = div(err[j], serr[j], lr, zero);
                let (expb, sexpb) = sub(
                    masked_cur(idx_b) - mask_cur(idx_b),
                    masked_cur(idx_bs) - mask_cur(idx_bs),
                    t, st,
                );

                result[idx_b]  =
                    masked_nxt(idx_b)  - (expb  + mask_nxt(idx_b));
                result[idx_bs] =
                    masked_nxt(idx_bs) - (sexpb + mask_nxt(idx_bs));
                result[idx_bs] +=
                    bit_constraint(masked_cur(idx_bs) - mask_cur(idx_bs));
            }
        } else {
            // For steps outside batch processing, maintain state (no constraints)
            for i in 0..result.len() {
                result[i] = E::ZERO;
            }
            
            // DEBUG: Log when no constraints are applied
            if current_step == self.pub_inputs.batch_size + 1 || 
               (current_step > self.pub_inputs.batch_size && current_step <= self.pub_inputs.batch_size + 5) {
                println!("DEBUG: No constraints applied for step {} (beyond batch_size {})", 
                         current_step, self.pub_inputs.batch_size);
            }
        }
    }

    fn context(&self) -> &AirContext<Felt> {
        &self.ctx
    }
}