// src/training/air.rs

use crate::helper::{f64_to_felt, bit_constraint};
use crate::signed::{add_generic as add, sub_generic as sub,
    mul_generic as mul, div_generic as div};

use winterfell::{
    Air, AirContext, Assertion, ByteWriter, EvaluationFrame, ProofOptions,
    TraceInfo, TransitionConstraintDegree,
};
use winter_utils::Serializable;
use winterfell::math::{FieldElement, ToElements};
use winterfell::math::fields::f128::BaseElement as Felt;

/// Public inputs for the masked zk‑STARK of one SGD step.
/// We only assert masked state boundaries, so raw values remain hidden.
#[derive(Clone)]
pub struct TrainingUpdateInputs {
    /// masked initial flattened state [v0+s0, v1+s1, …]
    pub initial_masked: Vec<Felt>,
    /// masked final flattened state
    pub final_masked: Vec<Felt>,
    /// number of steps = trace_length − 1
    pub steps: usize,
    /// public features and labels
    pub x: Vec<Felt>,
    pub y: Vec<Felt>,
    /// public scalar params
    pub learning_rate: Felt,
    pub precision: Felt,
}

impl Serializable for TrainingUpdateInputs {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        // write masked initial, masked final, then steps
        for &v in &self.initial_masked {
            target.write(v);
        }
        for &v in &self.final_masked {
            target.write(v);
        }
        target.write(f64_to_felt(self.steps as f64));
    }
}

impl ToElements<Felt> for TrainingUpdateInputs {
    fn to_elements(&self) -> Vec<Felt> {
        let mut v = self.initial_masked.clone();
        v.extend(self.final_masked.iter());
        v.push(f64_to_felt(self.steps as f64));
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
        Self { ctx: AirContext::new(ti, degrees, width, opt), pub_inputs }
    }

    fn get_assertions(&self) -> Vec<Assertion<Felt>> {
        let width = self.ctx.trace_info().width() / 2; // half is masked state
        let n = self.ctx.trace_len() - 1;
        let mut assertions = Vec::with_capacity(width * 2);
        // assert masked initial at row 0
        for i in 0..width {
            assertions.push(Assertion::single(i, 0, self.pub_inputs.initial_masked[i]));
        }
        // assert masked final at row n
        for i in 0..width {
            assertions.push(Assertion::single(i, n, self.pub_inputs.final_masked[i]));
        }
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
        let fe = self.pub_inputs.x.len();
        let ac = self.pub_inputs.y.len();
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

        // forward pass: compute raw errors
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
                let (p, sp) = mul(raw_v, raw_s, E::from(self.pub_inputs.x[i]), zero);
                let (nd, snd) = add(dot, sdot, p, sp);
                dot = nd; sdot = snd;
            }
            let (q, sdiv) = div(dot, sdot, pr, zero);

            let raw_b  = masked_cur(idx_b) - mask_cur(idx_b);
            let raw_bs = masked_cur(idx_bs) - mask_cur(idx_bs);
            let (pred, spred) = add(q, sdiv, raw_b, raw_bs);

            let (basic, sbasic) = sub(pred, spred, E::from(self.pub_inputs.y[j]), zero);
            let (num, snum)     = mul(basic, sbasic, two, zero);
            let (scaled, sscaled) =
                div(num, snum, E::from(f64_to_felt(ac as f64)), zero);
            err[j] = scaled; serr[j] = sscaled;
        }

        // backward pass + constraints
        for j in 0..ac {
            for i in 0..fe {
                let idx_v = 2 * (j * fe + i);
                let idx_s = idx_v + 1;

                let (p, sp) = mul(err[j], serr[j], E::from(self.pub_inputs.x[i]), zero);
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
    }

    fn context(&self) -> &AirContext<Felt> {
        &self.ctx
    }
}
