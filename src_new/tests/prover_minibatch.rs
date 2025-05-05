// tests/prover_minibatch.rs
use winterfell::{
    Air, EvaluationFrame, FieldElement, ToElements,
    crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree},
    AirContext, Assertion, ProofOptions, Serializable, StarkDomain, TraceInfo, TransitionConstraintDegree,
};
use winterfell::math::fields::f128::BaseElement as Felt;
use crate::training::{
    air::{TrainingUpdateAir, TrainingUpdateInputs},
    prover::TrainingUpdateProver,
};

// 1) A tiny “toy” dataset: 1 step, 2 features, 2 classes, batch_size = 1
fn make_toy() -> (
    Vec<Vec<Felt>>, Vec<Felt>,               // init_w, init_b
    Vec<Vec<Felt>>, Vec<Felt>,               // w_sign, b_sign
    Vec<Felt>, Vec<Felt>, Vec<Felt>, Vec<Felt>, // x, x_sign, y, y_sign
    Felt, Felt, ProofOptions,
) {
    let proof_options = ProofOptions::new(32, 8, 8, 8);

    // model: AC=2 classes, FE=2 features
    let init_w      = vec![ vec![Felt::ONE, Felt::ONE], vec![Felt::ONE, Felt::ONE] ];
    let init_b      = vec![ Felt::ZERO, Felt::ZERO ];
    let w_sign      = vec![ vec![Felt::ZERO, Felt::ZERO], vec![Felt::ZERO, Felt::ZERO] ];
    let b_sign      = vec![ Felt::ZERO, Felt::ZERO ];

    // batch_size = 1 => x.len()==FE, y.len()==AC
    let x      = vec![ Felt::ONE, Felt::ONE ];
    let x_sign = vec![ Felt::ZERO, Felt::ZERO ];
    let y      = vec![ Felt::ONE, Felt::ZERO ];
    let y_sign = vec![ Felt::ZERO, Felt::ZERO ];

    let lr = Felt::new(1u128);
    let pr = Felt::new(1u128);

    (init_w, init_b, w_sign, b_sign, x, x_sign, y, y_sign, lr, pr, proof_options)
}

#[test]
fn test_toy_minibatch_stark() {
    let (init_w, init_b, w_sign, b_sign,
         x, x_sign, y, y_sign,
         lr, pr, proof_options) = make_toy();

    // --- build prover and trace ---
    let prover = TrainingUpdateProver::new(
        proof_options.clone(),
        init_w.clone(), init_b.clone(),
        w_sign.clone(),  b_sign.clone(),
        x.clone(),      x_sign.clone(),
        y.clone(),      y_sign.clone(),
        lr, pr,
    );
    let trace = prover.build_trace();

    // --- manually walk the trace & check every transition step ---
    let air = TrainingUpdateAir::new(
        TraceInfo::new(trace.length(), trace.width()),
        prover.get_pub_inputs(&trace),
        proof_options.clone(),
    );

    let domain = StarkDomain::new(
        trace.length(),
        &air.context().proof_options().to_fri_options()
    );
    let mut frame = EvaluationFrame::new(trace.width());

    for step in 0..(trace.length() - 1) {
        // copy row `step`  into frame.current
        // copy row `step+1` into frame.next
        frame.current_mut().copy_from_slice(&trace.to_matrix()[step]);
        frame.next_mut().copy_from_slice(&trace.to_matrix()[step + 1]);

        let mut results = vec![ Felt::ZERO; trace.width() ];
        air.evaluate_transition(&frame, &[], &mut results);

        // at step 0 our guard does nothing.  from step=1 onward,
        // every result[k] MUST be zero.
        if step > 0 {
            for (i, r) in results.iter().enumerate() {
                assert_eq!(*r, Felt::ZERO,
                    "constraint {} failed at step {}", i, step);
            }
        }
    }

    // --- finally: full prove() / verify() roundtrip ---
    let st = std::time::Instant::now();
    let proof = prover.prove(trace.clone()).expect("prover failed");
    println!("toy proof gen: {:?}", st.elapsed());
    let pub_inputs = prover.get_pub_inputs(&trace);

    winterfell::verify::<
        TrainingUpdateAir,
        Blake3_256<Felt>,
        DefaultRandomCoin<Blake3_256<Felt>>,
        MerkleTree<Blake3_256<Felt>>
    >(proof, pub_inputs.clone(), &[]).expect("stark verify failed");
}
