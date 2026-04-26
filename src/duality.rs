use std::collections::HashMap;

use crate::multivector::MultiVector;
use crate::ops::{interior_left, interior_right, inverse, wedge};
use crate::vector::{Vector, pseudoscalar};

// =============================================================================
// Complements (Hodge Duality)
// =============================================================================

/// Right complement (Hodge dual): ⋆u = u ⌋ I⁻¹
///
/// Maps a grade-k element to grade-(d-k) by contracting with the inverse
/// pseudoscalar. In Euclidean 3D: vectors ↦ bivectors, bivectors ↦ vectors,
/// scalars ↦ pseudoscalar.
///
/// The right complement satisfies u ∧ ⋆u = ⟨u, u⟩ I.
pub fn right_complement<const D: usize>(u: &Vector<D>) -> Vector<D> {
    let ps = pseudoscalar(u.metric);
    let ps_inv = inverse(&ps).expect("pseudoscalar must be invertible (non-degenerate metric)");

    interior_left(u, &ps_inv)
}

/// Left complement: ⋆u = I⁻¹ ⌊ u
///
/// Maps a grade-k element to grade-(d-k) by contracting the inverse
/// pseudoscalar with u from the right. Differs from the right complement
/// by a sign that depends on grade and dimension.
pub fn left_complement<const D: usize>(u: &Vector<D>) -> Vector<D> {
    let ps = pseudoscalar(u.metric);
    let ps_inv = inverse(&ps).expect("pseudoscalar must be invertible (non-degenerate metric)");

    interior_right(&ps_inv, u)
}

/// Right complement of a multivector, applied grade by grade.
pub fn right_complement_mv<const D: usize>(m: &MultiVector<D>) -> MultiVector<D> {
    let mut components = HashMap::new();

    for (&k, component) in m.components() {
        let dual = right_complement(component);
        if !dual.is_zero(1e-15) {
            components.insert(D - k, dual);
        }
    }

    MultiVector::from_components(components, m.metric)
}

/// Left complement of a multivector, applied grade by grade.
pub fn left_complement_mv<const D: usize>(m: &MultiVector<D>) -> MultiVector<D> {
    let mut components = HashMap::new();

    for (&k, component) in m.components() {
        let dual = left_complement(component);
        if !dual.is_zero(1e-15) {
            components.insert(D - k, dual);
        }
    }

    MultiVector::from_components(components, m.metric)
}

// =============================================================================
// Join and Meet
// =============================================================================

/// Join of two blades: the smallest subspace containing both.
///
/// For blades with non-overlapping subspaces, this is the wedge product.
/// For overlapping subspaces, uses the dual:
///
/// ```text
/// join(A, B) = ⋆⁻¹(⋆A ∧ ⋆B)
/// ```
///
/// where ⋆ is the right complement.
pub fn join<const D: usize>(a: &Vector<D>, b: &Vector<D>) -> Vector<D> {
    // Try the wedge first — if it's nonzero, the subspaces are independent
    let w = wedge(a, b);
    if !w.is_zero(1e-15) {
        return w;
    }

    // Overlapping subspaces: use the dual route
    let a_dual = right_complement(a);
    let b_dual = right_complement(b);
    let meet_dual = wedge(&a_dual, &b_dual);

    // Undual the result
    let ps = pseudoscalar(a.metric);
    let ps_inv = inverse(&ps).expect("pseudoscalar must be invertible");

    interior_left(&meet_dual, &ps_inv)
}

/// Meet of two blades: their common subspace (intersection).
///
/// ```text
/// meet(A, B) = ⋆(⋆A ∧ ⋆B)
/// ```
///
/// Implemented via duality: dualize both, wedge, then dualize back.
/// Returns a zero vector if the blades share no common subspace.
pub fn meet<const D: usize>(a: &Vector<D>, b: &Vector<D>) -> Vector<D> {
    let a_dual = right_complement(a);
    let b_dual = right_complement(b);
    let wedged = wedge(&a_dual, &b_dual);

    right_complement(&wedged)
}
