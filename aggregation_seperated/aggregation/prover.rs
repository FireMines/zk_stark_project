use crate::aggregation::air::{GlobalUpdateAir, GlobalUpdateInputs};
use crate::helper::{AC, FE, f64_to_felt, transpose};
use winterfell::{
    AuxRandElements, CompositionPoly, CompositionPolyTrace, ConstraintCompositionCoefficients,
    DefaultConstraintCommitment, DefaultConstraintEvaluator, DefaultTraceLde, PartitionOptions, ProofOptions,
    Prover, TraceInfo, TraceTable, TracePolyTable,
};
use winterfell::crypto::{hashers::Blake3_256, DefaultRandomCoin, MerkleTree};
use winterfell::math::FieldElement;
use winterfell::math::fields::f128::BaseElement as Felt;

pub struct GlobalUpdateProver {
    pub options: ProofOptions,
    pub global_w: Vec<Vec<Felt>>,
    pub global_b: Vec<Felt>,
    pub local_w: Vec<Vec<Vec<Felt>>>, // [C][AC][FE]
    pub local_b: Vec<Vec<Felt>>,      // [C][AC]
    pub k: Felt,                    // scaling factor = number of clients
    pub trace_length: usize,        // padded length (power-of-two, at least uns_padded_steps)
}

impl GlobalUpdateProver {
    pub fn new(
        options: ProofOptions,
        global_w: Vec<Vec<Felt>>,
        global_b: Vec<Felt>,
        local_w: Vec<Vec<Vec<Felt>>>,
        local_b: Vec<Vec<Felt>>,
        k: Felt,
    ) -> Self {
        // We now set uns_padded_steps to: 1 (initial row) + number of client updates + 1 extra final row.
        let uns_padded_steps = local_w.len() + 2;
        let padded_rows = uns_padded_steps.next_power_of_two().max(8);
        Self {
            options,
            global_w,
            global_b,
            local_w,
            local_b,
            k,
            trace_length: padded_rows,
        }
    }

    /// Flatten state: weights then biases.
    fn flatten_state(w: &Vec<Vec<Felt>>, b: &Vec<Felt>) -> Vec<Felt> {
        let mut flat = Vec::new();
        for row in w {
            flat.extend_from_slice(row);
        }
        flat.extend_from_slice(b);
        flat
    }

    /// Build an augmented iterative trace.
    ///
    /// We "anchor" each client update to the fixed base state S₀ (the initial global model).
    /// For each client update L, we compute:
    ///
    ///  δ = (L – S₀)/c,
    ///
    /// and update the accumulated state by:
    ///
    ///  S_next = S_current + δ.
    ///
    /// Each row is augmented as [S ‖ (L – S₀)].
    /// Row 0 is [S₀ ‖ 0]. After processing all client updates, we add an extra row
    /// [S_final ‖ 0] to serve as the final boundary.
    pub fn compute_iterative_trace_augmented(&self) -> Vec<Vec<Felt>> {
        let num_transitions = self.local_w.len();
        // uns_padded_steps = 1 (row 0) + num_transitions + 1 (extra final row)
        let uns_padded_steps = num_transitions + 2;
        let d_state = AC * FE + AC;
        let mut trace_rows: Vec<Vec<Felt>> = Vec::with_capacity(uns_padded_steps);

        // S₀ = initial global state.
        let base_state = Self::flatten_state(&self.global_w, &self.global_b);
        // Row 0: [S₀ ‖ 0]
        let mut row0 = base_state.clone();
        row0.extend(vec![Felt::ZERO; d_state]);
        trace_rows.push(row0);

        let c_val = self.k;
        // Start with current state = S₀.
        let mut current_state = base_state.clone();

        // For each client update:
        for client in 0..num_transitions {
            // Compute L: client’s local model (flatten).
            let local_w_flat: Vec<Felt> = self.local_w[client]
                .iter()
                .flat_map(|row| row.clone())
                .collect();
            let mut L = local_w_flat;
            L.extend(self.local_b[client].clone());
            // Compute δ = (L – S₀)/c.
            let delta: Vec<Felt> = base_state
                .iter()
                .zip(L.iter())
                .map(|(b, l)| (*l - *b) / c_val)
                .collect();
            // Compute next state: S_next = current_state + δ.
            let next_state: Vec<Felt> = current_state
                .iter()
                .zip(delta.iter())
                .map(|(s, d)| *s + *d)
                .collect();
            // In this row, store update U = (L – S₀).
            let update_stored: Vec<Felt> = base_state
                .iter()
                .zip(L.iter())
                .map(|(b, l)| *l - *b)
                .collect();
            let mut augmented_row = next_state.clone();
            augmented_row.extend(update_stored);
            trace_rows.push(augmented_row);
            current_state = next_state;
        }

        // Add an extra row: [S_final ‖ 0] where S_final = current_state.
        let extra_row = [current_state.clone(), vec![Felt::ZERO; d_state]].concat();
        trace_rows.push(extra_row);

        // Pad to reach trace_length.
        while trace_rows.len() < self.trace_length {
            let last_state = trace_rows.last().unwrap()[..d_state].to_vec();
            let padded_row = [last_state, vec![Felt::ZERO; d_state]].concat();
            trace_rows.push(padded_row);
        }
        trace_rows
    }

    /// Build the execution trace as a column-major matrix.
    pub fn build_trace(&self) -> TraceTable<Felt> {
        let trace_rows = self.compute_iterative_trace_augmented();
        TraceTable::init(transpose(trace_rows))
    }

    /// Retrieve public inputs.
    ///
    /// The final aggregated state is taken from the S part of the final uns‑padded row.
    pub fn get_pub_inputs(&self, _trace: &<GlobalUpdateProver as Prover>::Trace) -> GlobalUpdateInputs {
        // uns_padded_steps = local_w.len() + 2.
        let uns_padded_steps = self.local_w.len() + 2;
        let trace_rows = self.compute_iterative_trace_augmented();
        let final_row = trace_rows[uns_padded_steps - 1].clone();
        let d_state = AC * FE + AC;
        let new_global_w: Vec<Vec<Felt>> = final_row[..(AC * FE)]
            .chunks(FE)
            .map(|chunk| chunk.to_vec())
            .collect();
        let new_global_b: Vec<Felt> = final_row[(AC * FE)..d_state].to_vec();
        GlobalUpdateInputs {
            global_w: self.global_w.clone(),
            global_b: self.global_b.clone(),
            new_global_w,
            new_global_b,
            k: self.k,
            digest: f64_to_felt(0.0),
            steps: uns_padded_steps,
        }
    }
}

impl Prover for GlobalUpdateProver {
    type BaseField = Felt;
    type Air = GlobalUpdateAir;
    type Trace = TraceTable<Felt>;
    type HashFn = Blake3_256<Felt>;
    type VC = MerkleTree<Self::HashFn>;
    type RandomCoin = DefaultRandomCoin<Self::HashFn>;
    type TraceLde<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultTraceLde<E, Self::HashFn, Self::VC>;
    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, Self::Air, E>;
    type ConstraintCommitment<E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintCommitment<E, Self::HashFn, Self::VC>;

    fn get_pub_inputs(&self, trace: &Self::Trace) -> GlobalUpdateInputs {
        self.get_pub_inputs(trace)
    }

    fn options(&self) -> &ProofOptions {
        &self.options
    }

    fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &winterfell::matrix::ColMatrix<Self::BaseField>,
        domain: &winterfell::StarkDomain<Self::BaseField>,
        partition_options: PartitionOptions,
    ) -> (Self::TraceLde<E>, winterfell::TracePolyTable<E>) {
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
        domain: &winterfell::StarkDomain<Self::BaseField>,
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
