use crate::multivector::MultiVector;
use crate::vector::Vector;

// =============================================================================
// Exponential
// =============================================================================

/// Exponential of a scalar or bivector.
///
/// For a scalar s: exp(s) = e^s (the ordinary exponential).
///
/// For a bivector B, the result depends on B²:
///
/// **Spacelike** (B² < 0, Euclidean planes): circular exponential
/// ```text
/// exp(B) = cos θ + B̂ sin θ,    θ = √(-B²)
/// ```
///
/// **Timelike** (B² > 0, Lorentzian boosts): hyperbolic exponential
/// ```text
/// exp(B) = cosh θ + B̂ sinh θ,  θ = √(B²)
/// ```
///
/// **Null** (B² = 0, degenerate planes): nilpotent
/// ```text
/// exp(B) = 1 + B
/// ```
///
/// For the zero bivector, exp(0) = 1.
pub fn exp<const D: usize>(v: &Vector<D>) -> MultiVector<D> {
    let k = v.grade();

    match k {
        0 => {
            let s = v.data[ndarray::IxDyn(&[])];
            let result = Vector::scalar(s.exp(), v.metric);

            MultiVector::from_vector(result)
        }

        2 => exp_bivector(v),

        _ => panic!(
            "exp is implemented for grade 0 (scalar) and grade 2 (bivector), got grade {}",
            k
        ),
    }
}

/// Exponential of a bivector, branching on B².
///
/// norm_squared() returns <B ~B>_0 = -B² for grade 2, so:
/// - norm_squared > 0 → B² < 0 → spacelike → circular
/// - norm_squared < 0 → B² > 0 → timelike → hyperbolic
/// - norm_squared ≈ 0 → null → nilpotent
fn exp_bivector<const D: usize>(b: &Vector<D>) -> MultiVector<D> {
    let ns = b.norm_squared(); // = -B²

    if ns.abs() < 1e-15 {
        // Null or zero bivector: exp(B) = 1 + B
        let scalar = Vector::scalar(1.0, b.metric);
        if b.is_zero(1e-15) {
            return MultiVector::from_vector(scalar);
        }

        return MultiVector::from_vector(scalar) + MultiVector::from_vector(b.clone());
    }

    if ns > 0.0 {
        // Spacelike (B² < 0): circular
        let theta = ns.sqrt();
        let b_unit = b / theta;
        let scalar = Vector::scalar(theta.cos(), b.metric);
        let bivector = &b_unit * theta.sin();

        MultiVector::from_vector(scalar) + MultiVector::from_vector(bivector)
    } else {
        // Timelike (B² > 0): hyperbolic
        let theta = (-ns).sqrt();
        let b_unit = b / theta;
        let scalar = Vector::scalar(theta.cosh(), b.metric);
        let bivector = &b_unit * theta.sinh();

        MultiVector::from_vector(scalar) + MultiVector::from_vector(bivector)
    }
}

// =============================================================================
// Logarithm
// =============================================================================

/// Logarithm of a rotor (even-grade multivector).
///
/// For a **circular** rotor R = cos θ + sin θ B̂ (scalar part in [-1, 1]):
/// ```text
/// log(R) = θ B̂,    θ = arccos(scalar part)
/// ```
///
/// For a **hyperbolic** rotor R = cosh θ + sinh θ B̂ (|scalar part| > 1):
/// ```text
/// log(R) = θ B̂,    θ = acosh(scalar part)
/// ```
///
/// The result is a bivector. This is the inverse of `exp` restricted to
/// bivectors: exp(log(R)) = R.
///
/// For the identity rotor (θ = 0), returns the zero bivector.
pub fn log<const D: usize>(r: &MultiVector<D>) -> Vector<D> {
    let a = r.scalar_part();
    let bivector_part = r.grade_project(2);

    if a.abs() <= 1.0 + 1e-12 {
        // Circular rotor: scalar = cos θ
        let cos_clamped = a.clamp(-1.0, 1.0);
        let theta = cos_clamped.acos();
        let sin_theta = theta.sin();

        if sin_theta.abs() < 1e-15 {
            // Near identity or near π: return bivector part as-is
            return bivector_part;
        }

        &bivector_part * (theta / sin_theta)
    } else {
        // Hyperbolic rotor: scalar = ±cosh θ
        let theta = a.abs().acosh();
        let sinh_theta = theta.sinh();

        if sinh_theta.abs() < 1e-15 {
            return bivector_part;
        }

        // Sign: if scalar is negative, the generator picks up a π rotation
        // component. For the standard case (a > 1), sign is +1.
        let sign = a.signum();

        &bivector_part * (sign * theta / sinh_theta)
    }
}

// =============================================================================
// Spherical Linear Interpolation
// =============================================================================

/// Spherical linear interpolation between two rotors.
///
/// ```text
/// slerp(R0, R1, t) = R0 exp(t · log(R0^{-1} R1))
/// ```
///
/// At t = 0 returns R0, at t = 1 returns R1. The interpolation follows
/// the shortest arc on the rotor manifold.
///
/// Both R0 and R1 should be unit rotors (R ~R = 1).
pub fn slerp<const D: usize>(r0: &MultiVector<D>, r1: &MultiVector<D>, t: f64) -> MultiVector<D> {
    use crate::ops::geometric_mv_mv;

    // R0^{-1} = ~R0 for unit rotors
    let r0_inv = r0.rev();

    // Relative rotor: R0^{-1} R1
    let relative = geometric_mv_mv(&r0_inv, r1);

    // Extract generator and scale by t
    let generator = log(&relative);
    let scaled = &generator * t;

    // R0 * exp(t * generator)
    let interpolated = exp(&scaled);

    geometric_mv_mv(r0, &interpolated)
}
