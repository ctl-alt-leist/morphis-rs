use crate::multivector::MultiVector;
use crate::ops::{geometric_mv_mv, geometric_mv_v};
use crate::vector::Vector;

// =============================================================================
// Sandwich Product
// =============================================================================

/// Transform a k-vector via the sandwich product: M v ~M.
///
/// The result is grade-projected back to the input grade. This is the
/// standard action of a versor on an element of the algebra — rotors
/// rotate, vectors reflect, motors translate (in PGA).
pub fn transform<const D: usize>(v: &Vector<D>, m: &MultiVector<D>) -> Vector<D> {
    let m_rev = m.rev();
    let temp = geometric_mv_v(m, v);
    let result = geometric_mv_mv(&temp, &m_rev);

    result.grade_project(v.grade())
}

/// Transform a multivector via the sandwich product: M N ~M.
///
/// Each grade component of the input is independently grade-projected
/// in the result.
pub fn transform_mv<const D: usize>(n: &MultiVector<D>, m: &MultiVector<D>) -> MultiVector<D> {
    let m_rev = m.rev();
    let temp = geometric_mv_mv(m, n);
    let result = geometric_mv_mv(&temp, &m_rev);

    let mut components = std::collections::HashMap::new();
    for &k in &n.grades() {
        let projected = result.grade_project(k);
        if !projected.is_zero(1e-15) {
            components.insert(k, projected);
        }
    }

    MultiVector::from_components(components, m.metric)
}

// =============================================================================
// Constructors
// =============================================================================

/// Create a rotor from a bivector plane and rotation angle.
///
/// The rotor is the multivector M = cos(θ/2) - sin(θ/2) B̂ where B̂ is
/// the unit bivector in the plane of rotation. Apply via the sandwich
/// product, either explicitly or through `transform`:
///
/// ```text
/// v' = M v ~M          // explicit
/// v' = transform(v, M) // with grade projection
/// ```
pub fn rotor<const D: usize>(plane: &Vector<D>, angle: f64) -> MultiVector<D> {
    assert_eq!(
        plane.grade(),
        2,
        "rotor requires a grade-2 bivector, got grade {}",
        plane.grade()
    );

    let plane_unit = plane
        .normalize()
        .expect("rotor plane bivector must be nonzero");

    let half = angle / 2.0;
    let scalar = Vector::scalar(half.cos(), plane.metric);
    let bivector = &plane_unit * (-half.sin());

    MultiVector::from_vector(scalar) + MultiVector::from_vector(bivector)
}
