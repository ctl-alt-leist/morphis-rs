use crate::multivector::MultiVector;
use crate::ops::{geometric_mv_mv, geometric_mv_v};
use crate::vector::Vector;

/// A versor: an invertible multivector that acts on elements via the
/// sandwich product v ↦ M v ~M.
///
/// Versors are the native transformations of geometric algebra. Rotors
/// (even-grade versors) encode rotations; single vectors encode reflections;
/// motors (in PGA) encode rigid motions. Composition of versors is the
/// geometric product of their motors.
#[derive(Debug, Clone)]
pub struct Versor<const D: usize> {
    motor: MultiVector<D>,
    motor_rev: MultiVector<D>,
}

impl<const D: usize> Versor<D> {
    /// Create a versor from a multivector.
    ///
    /// The multivector must be invertible (nonzero norm). The reverse is
    /// cached for efficient repeated application.
    pub fn from_multivector(mv: MultiVector<D>) -> Self {
        let motor_rev = mv.rev();

        Self {
            motor: mv,
            motor_rev,
        }
    }

    /// Transform a k-vector via the sandwich product: M v ~M.
    ///
    /// The result is grade-projected back to the input grade, preserving
    /// the grade of the operand.
    pub fn transform(&self, v: &Vector<D>) -> Vector<D> {
        let temp = geometric_mv_v(&self.motor, v);
        let result = geometric_mv_mv(&temp, &self.motor_rev);

        result.grade_project(v.grade())
    }

    /// Transform a multivector via the sandwich product: M N ~M.
    ///
    /// Each grade component is independently grade-projected.
    pub fn transform_mv(&self, n: &MultiVector<D>) -> MultiVector<D> {
        let temp = geometric_mv_mv(&self.motor, n);
        let result = geometric_mv_mv(&temp, &self.motor_rev);

        // Grade-project: keep only the grades that were present in the input
        let mut components = std::collections::HashMap::new();
        for &k in &n.grades() {
            let projected = result.grade_project(k);
            if !projected.is_zero(1e-15) {
                components.insert(k, projected);
            }
        }

        MultiVector::from_components(components, self.motor.metric)
    }

    /// Compose two versors: the result transforms as (self ∘ other),
    /// meaning other is applied first, then self.
    ///
    /// Composition is the geometric product of the motors.
    pub fn compose(&self, other: &Versor<D>) -> Versor<D> {
        let motor = geometric_mv_mv(&self.motor, &other.motor);

        Versor::from_multivector(motor)
    }

    /// Inverse versor: undoes the transformation.
    pub fn inv(&self) -> Versor<D> {
        Versor {
            motor: self.motor_rev.clone(),
            motor_rev: self.motor.clone(),
        }
    }

    /// Access the underlying multivector (the motor).
    pub fn motor(&self) -> &MultiVector<D> {
        &self.motor
    }

    /// Whether this is an even-grade versor (rotor, motor).
    pub fn is_even(&self) -> bool {
        self.motor.is_even()
    }

    /// Dimension of the underlying vector space.
    pub fn dim(&self) -> usize {
        D
    }
}

// =============================================================================
// Constructors
// =============================================================================

/// Create a rotor from a bivector plane and rotation angle.
///
/// The rotor is M = cos(θ/2) - sin(θ/2) B̂ where B̂ is the unit bivector
/// in the plane of rotation. Transforms via the sandwich product:
///
/// ```text
/// v' = M v ~M
/// ```
///
/// rotates v by angle θ in the plane of B.
pub fn rotor<const D: usize>(plane: &Vector<D>, angle: f64) -> Versor<D> {
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

    let motor = MultiVector::from_vector(scalar) + MultiVector::from_vector(bivector);

    Versor::from_multivector(motor)
}

// =============================================================================
// Operator Overloads
// =============================================================================

// Sandwich product: Versor % Vector -> Vector
impl<const D: usize> std::ops::Rem<&Vector<D>> for &Versor<D> {
    type Output = Vector<D>;
    fn rem(self, rhs: &Vector<D>) -> Vector<D> {
        self.transform(rhs)
    }
}

impl<const D: usize> std::ops::Rem<Vector<D>> for Versor<D> {
    type Output = Vector<D>;
    fn rem(self, rhs: Vector<D>) -> Vector<D> {
        self.transform(&rhs)
    }
}

impl<const D: usize> std::ops::Rem<Vector<D>> for &Versor<D> {
    type Output = Vector<D>;
    fn rem(self, rhs: Vector<D>) -> Vector<D> {
        self.transform(&rhs)
    }
}

impl<const D: usize> std::ops::Rem<&Vector<D>> for Versor<D> {
    type Output = Vector<D>;
    fn rem(self, rhs: &Vector<D>) -> Vector<D> {
        self.transform(rhs)
    }
}

// Sandwich product: Versor % MultiVector -> MultiVector
impl<const D: usize> std::ops::Rem<&MultiVector<D>> for &Versor<D> {
    type Output = MultiVector<D>;
    fn rem(self, rhs: &MultiVector<D>) -> MultiVector<D> {
        self.transform_mv(rhs)
    }
}

impl<const D: usize> std::ops::Rem<MultiVector<D>> for Versor<D> {
    type Output = MultiVector<D>;
    fn rem(self, rhs: MultiVector<D>) -> MultiVector<D> {
        self.transform_mv(&rhs)
    }
}

impl<const D: usize> std::ops::Rem<MultiVector<D>> for &Versor<D> {
    type Output = MultiVector<D>;
    fn rem(self, rhs: MultiVector<D>) -> MultiVector<D> {
        self.transform_mv(&rhs)
    }
}

impl<const D: usize> std::ops::Rem<&MultiVector<D>> for Versor<D> {
    type Output = MultiVector<D>;
    fn rem(self, rhs: &MultiVector<D>) -> MultiVector<D> {
        self.transform_mv(rhs)
    }
}

// Composition: Versor * Versor -> Versor
impl<const D: usize> std::ops::Mul for &Versor<D> {
    type Output = Versor<D>;
    fn mul(self, rhs: &Versor<D>) -> Versor<D> {
        self.compose(rhs)
    }
}

impl<const D: usize> std::ops::Mul for Versor<D> {
    type Output = Versor<D>;
    fn mul(self, rhs: Versor<D>) -> Versor<D> {
        self.compose(&rhs)
    }
}

impl<const D: usize> std::ops::Mul<Versor<D>> for &Versor<D> {
    type Output = Versor<D>;
    fn mul(self, rhs: Versor<D>) -> Versor<D> {
        self.compose(&rhs)
    }
}

impl<const D: usize> std::ops::Mul<&Versor<D>> for Versor<D> {
    type Output = Versor<D>;
    fn mul(self, rhs: &Versor<D>) -> Versor<D> {
        self.compose(rhs)
    }
}

// Inverse: !Versor -> Versor
impl<const D: usize> std::ops::Not for &Versor<D> {
    type Output = Versor<D>;
    fn not(self) -> Versor<D> {
        self.inv()
    }
}

impl<const D: usize> std::ops::Not for Versor<D> {
    type Output = Versor<D>;
    fn not(self) -> Versor<D> {
        self.inv()
    }
}
