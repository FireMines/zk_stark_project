use winterfell::math::{FieldElement, fields::f128::BaseElement as Felt};

pub const MAX: Felt = Felt::new(u128::MAX);
pub const THRESHOLD: Felt =
    Felt::new(170_141_183_460_469_231_731_687_303_715_884_105_727);

// ----------------------------------------------------------------------
// generic helpers ------------------------------------------------------

#[inline(always)]
fn cleanse<E: FieldElement + From<Felt>>(v: E, s: E) -> E {
    let max = E::from(MAX);
    (E::ONE - s) * v + s * (max - v + E::ONE)
}

#[inline(always)]
pub fn add_generic<E: FieldElement + From<Felt>>(a: E, s_a: E, b: E, s_b: E) -> (E, E) {
    let max = E::from(MAX);
    let a_c = cleanse(a, s_a);
    let b_c = cleanse(b, s_b);
    let ind = s_a * s_b;                       // both negative?
    let c   = ind * (max + E::ONE - a_c - b_c) // wrapped
            + (E::ONE - ind) * (a + b);        // normal
    (c, ind)                                   // sign == ind
}

#[inline(always)]
pub fn sub_generic<E: FieldElement + From<Felt>>(a: E, s_a: E, b: E, s_b: E) -> (E, E) {
    add_generic(a, s_a, b, E::ONE - s_b)       // a + (-b)
}

#[inline(always)]
pub fn mul_generic<E: FieldElement + From<Felt>>(a: E, s_a: E, b: E, s_b: E) -> (E, E) {
    let max = E::from(MAX);
    let prod = cleanse(a, s_a) * cleanse(b, s_b);
    let sign = s_a + s_b - s_a * s_b * E::from(Felt::from(2u64)); // XOR
    let res  = sign * (max - prod + E::ONE) + (E::ONE - sign) * prod;
    (res, sign)
}

#[inline(always)]
pub fn div_generic<E: FieldElement + From<Felt>>(a: E, s_a: E, b: E, s_b: E) -> (E, E) {
    let max  = E::from(MAX);
    let q    = cleanse(a, s_a) * cleanse(b, s_b).inv();
    let sign = s_a + s_b - s_a * s_b * E::from(Felt::from(2u64)); // XOR
    let res  = sign * (max + E::ONE - q) + (E::ONE - sign) * q;
    (res, sign)
}

// ----------------------------------------------------------------------
// felt‑only wrappers for the prover ------------------------------------

pub fn add(a: Felt, b: Felt, s_a: Felt, s_b: Felt) -> (Felt, Felt) {
    add_generic(a, s_a, b, s_b)
}
pub fn sub(a: Felt, b: Felt, s_a: Felt, s_b: Felt) -> (Felt, Felt) {
    sub_generic(a, s_a, b, s_b)
}
pub fn mul(a: Felt, b: Felt, s_a: Felt, s_b: Felt) -> (Felt, Felt) {
    mul_generic(a, s_a, b, s_b)
}
pub fn div(a: Felt, b: Felt, s_a: Felt, s_b: Felt) -> (Felt, Felt) {
    div_generic(a, s_a, b, s_b)
}
