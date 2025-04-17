// src/training/air.rs
//
// STARK AIR for one SGD step with explicit sign columns.
// Layout per weight/bias cell:  [value, sign]  (sign ∈ {0,1})
// ─────────────────────────────────────────────────────────────

use crate::helper::{f64_to_felt, bit_constraint};
// training/air.rs
use crate::signed::{add_generic as add, sub_generic as sub,
    mul_generic as mul, div_generic as div};

use winterfell::{
    Air, AirContext, Assertion, ByteWriter, EvaluationFrame, ProofOptions,
    TraceInfo, TransitionConstraintDegree,
};
use winter_utils::Serializable;
use winterfell::math::{FieldElement, ToElements};
use winterfell::math::fields::f128::BaseElement as Felt;

// -----------------------------------------------------------------------------
//  Signed arithmetic *algebraic* gadgets (no compare; sign is an input column)
// -----------------------------------------------------------------------------

#[inline]
fn to_e<E: FieldElement<BaseField = Felt>>(v: Felt) -> E { E::from(v) }



#[derive(Clone)]
pub struct TrainingUpdateInputs {
    pub initial: Vec<Felt>,
    pub final_state: Vec<Felt>,
    pub steps: usize,
    pub x: Vec<Felt>,
    pub y: Vec<Felt>,
    pub learning_rate: Felt,
    pub precision: Felt,
}

impl Serializable for TrainingUpdateInputs {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        for v in &self.initial { target.write(*v); }
        for v in &self.final_state { target.write(*v); }
        target.write(f64_to_felt(self.steps as f64));
    }
}
impl ToElements<Felt> for TrainingUpdateInputs {
    fn to_elements(&self) -> Vec<Felt> {
        let mut v = self.initial.clone();
        v.extend(self.final_state.clone());
        v.push(f64_to_felt(self.steps as f64));
        v
    }
}

// ------------------------------------------------------------ AIR ------------

pub struct TrainingUpdateAir {
    ctx: AirContext<Felt>,
    pub_inputs: TrainingUpdateInputs,
}
impl Air for TrainingUpdateAir {
    type BaseField = Felt;
    type PublicInputs = TrainingUpdateInputs;

    fn new(ti: TraceInfo, pub_inputs: TrainingUpdateInputs, opt: ProofOptions) -> Self {
        // every logical cell = [val, sign]  => width doubles
        let state_width = pub_inputs.initial.len();          // already doubled in prover
        let deg = vec![TransitionConstraintDegree::new(1); state_width];
        Self { ctx: AirContext::new(ti, deg, state_width, opt), pub_inputs }
    }

    #[allow(clippy::too_many_lines)]
    fn evaluate_transition<E: FieldElement<BaseField = Felt>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic: &[E],
        result: &mut [E],
    ) {
        let fe = self.pub_inputs.x.len();
        let ac = self.pub_inputs.y.len();
        let two = E::from(f64_to_felt(2.0));
        let pr  = E::from(self.pub_inputs.precision);
        let lr  = E::from(self.pub_inputs.learning_rate);
        let zero = E::ZERO;

        //---------------- forward pass ----------------------------------------
        // Hold (error, s_error) for each activation
        let mut err  = vec![zero; ac];
        let mut serr = vec![zero; ac];

        for j in 0..ac {
            // indices helpers
            let idx_w = |i| 2*(j*fe + i);         // value column
            let idx_ws = |i| idx_w(i)+1;          // sign column
            let idx_b  = 2*(ac*fe + j);
            let idx_bs = idx_b + 1;

            // dot = Σ w·x
            let mut dot = zero; let mut sdot = zero;
            for i in 0..fe {
                let (p, sp) = mul(
                    frame.current()[idx_w(i)],  frame.current()[idx_ws(i)],
                    E::from(self.pub_inputs.x[i]), zero
                );
                let (nd, snd) = add(dot, sdot, p, sp);
                dot = nd; sdot = snd;
            }
            let (q, sdiv) = div(dot, sdot, pr, zero);
            let (pred, spred) = add(q, sdiv,
                                    frame.current()[idx_b], frame.current()[idx_bs]);

            // error = (pred - y) * 2/ac
            let (basic, sbasic) = sub(pred, spred,
                                      E::from(self.pub_inputs.y[j]), zero);
            let (num, snum) = mul(basic, sbasic, two, zero);
            let (scaled, sscaled) = div(num, snum,
                                        E::from(f64_to_felt(ac as f64)), zero);
            err[j]  = scaled;  serr[j] = sscaled;
        }

        //---------------- backward pass ---------------------------------------
        for j in 0..ac {
            for i in 0..fe {
                let idx_v  = 2*(j*fe + i);
                let idx_s  = idx_v + 1;

                // grad = error * x / lr / pr
                let (p, sp)   = mul(err[j], serr[j],
                                    E::from(self.pub_inputs.x[i]), zero);
                let (t, st)   = div(p, sp, lr, zero);
                let (grad, sg)= div(t, st, pr, zero);

                let (exp, sexp) = sub(frame.current()[idx_v], frame.current()[idx_s],
                                      grad, sg);

                // constraint: next_val == exp  and  next_sign == sexp
                result[idx_v] = frame.next()[idx_v] - exp;
                result[idx_s] = frame.next()[idx_s] - sexp;

                // sign bit boolean constraint
                result[idx_s] += bit_constraint(frame.current()[idx_s]);
            }

            // bias cell
            let idx_b  = 2*(ac*fe + j);
            let idx_bs = idx_b + 1;

            let (t, st)  = div(err[j], serr[j], lr, zero);
            let (expb, sexpb) = sub(frame.current()[idx_b], frame.current()[idx_bs], t, st);

            result[idx_b]  = frame.next()[idx_b]  - expb;
            result[idx_bs] = frame.next()[idx_bs] - sexpb;
            result[idx_bs] += bit_constraint(frame.current()[idx_bs]);
        }
    }

    fn get_assertions(&self) -> Vec<Assertion<Felt>> {
        let n = self.trace_length() - 1;
        self.pub_inputs.final_state.iter().enumerate()
            .map(|(i,&v)| Assertion::single(i, n, v))
            .collect()
    }
    fn context(&self) -> &AirContext<Felt> { &self.ctx }
}
