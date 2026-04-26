use std::ops::Deref;

use crate::exponential::exp;
use crate::multivector::MultiVector;
use crate::ops::{geometric_mv_mv, geometric_mv_v, geometric_v_mv};
use crate::vector::Vector;

// =============================================================================
// Rotor Type
// =============================================================================

/// A rotor: an even-grade unit multivector with cached reverse.
///
/// A `Rotor` derefs to `MultiVector`, so it can be used anywhere a
/// multivector reference is expected — all existing operators and
/// functions work transparently. The cached reverse avoids recomputation
/// when the same rotor is applied to many elements.
#[derive(Debug, Clone)]
pub struct Rotor<const D: usize> {
    mv: MultiVector<D>,
    mv_rev: MultiVector<D>,
}

impl<const D: usize> Deref for Rotor<D> {
    type Target = MultiVector<D>;

    fn deref(&self) -> &MultiVector<D> {
        &self.mv
    }
}

impl<const D: usize> Rotor<D> {
    /// Cached reverse: returns a reference rather than recomputing.
    pub fn rev(&self) -> &MultiVector<D> {
        &self.mv_rev
    }
}

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
/// The rotor is M = cos(θ/2) - sin(θ/2) B̂ where B̂ is the unit bivector
/// in the plane of rotation. The reverse is cached at construction.
///
/// A `Rotor` derefs to `MultiVector`, so all multivector operators work
/// directly:
///
/// ```text
/// v' = &R * &v * R.rev()   // explicit sandwich product
/// v' = transform(&v, &R)   // with grade projection
/// R3 = &R2 * &R1           // composition via geometric product
/// ```
pub fn rotor<const D: usize>(plane: &Vector<D>, angle: f64) -> Rotor<D> {
    assert_eq!(
        plane.grade(),
        2,
        "rotor requires a grade-2 bivector, got grade {}",
        plane.grade()
    );

    let plane_unit = plane
        .normalize()
        .expect("rotor plane bivector must be nonzero");

    // R = exp(-B̂ θ/2)
    let generator = &plane_unit * (-angle / 2.0);
    let mv = exp(&generator);
    let mv_rev = mv.rev();

    Rotor { mv, mv_rev }
}

// =============================================================================
// Operator Overloads
// =============================================================================

// Deref coercion does not apply to binary operators in Rust, so we
// implement the geometric product for Rotor operands explicitly.

// Rotor * Rotor -> MultiVector (composition)
impl<const D: usize> std::ops::Mul for &Rotor<D> {
    type Output = MultiVector<D>;
    fn mul(self, rhs: &Rotor<D>) -> MultiVector<D> {
        geometric_mv_mv(&self.mv, &rhs.mv)
    }
}

// Rotor * Vector -> MultiVector (first half of sandwich)
impl<const D: usize> std::ops::Mul<&Vector<D>> for &Rotor<D> {
    type Output = MultiVector<D>;
    fn mul(self, rhs: &Vector<D>) -> MultiVector<D> {
        geometric_mv_v(&self.mv, rhs)
    }
}

// Vector * Rotor -> MultiVector
impl<const D: usize> std::ops::Mul<&Rotor<D>> for &Vector<D> {
    type Output = MultiVector<D>;
    fn mul(self, rhs: &Rotor<D>) -> MultiVector<D> {
        geometric_v_mv(self, &rhs.mv)
    }
}

// MultiVector * Rotor -> MultiVector (chaining composed products)
impl<const D: usize> std::ops::Mul<&Rotor<D>> for &MultiVector<D> {
    type Output = MultiVector<D>;
    fn mul(self, rhs: &Rotor<D>) -> MultiVector<D> {
        geometric_mv_mv(self, &rhs.mv)
    }
}

// Rotor * MultiVector -> MultiVector
impl<const D: usize> std::ops::Mul<&MultiVector<D>> for &Rotor<D> {
    type Output = MultiVector<D>;
    fn mul(self, rhs: &MultiVector<D>) -> MultiVector<D> {
        geometric_mv_mv(&self.mv, rhs)
    }
}
