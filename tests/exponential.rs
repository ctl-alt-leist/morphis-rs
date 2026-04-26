use std::f64::consts::PI;

use nalgebra::DMatrix;

use morphis::exponential::{exp, log, slerp};
use morphis::metric::{Metric, euclidean, lorentzian};
use morphis::operator::Operator;
use morphis::ops::wedge;
use morphis::vector::{Vector, basis};
use morphis::versor::{rotor, transform};

// =============================================================================
// exp: Scalar
// =============================================================================

#[test]
fn exp_scalar() {
    let g: Metric<3> = euclidean();

    let s = Vector::<3>::scalar(1.0, g);
    let result = exp(&s);

    let e = std::f64::consts::E;
    assert!(
        (result.scalar_part() - e).abs() < 1e-12,
        "exp(1) should be e, got {}",
        result.scalar_part(),
    );
}

#[test]
fn exp_zero_scalar() {
    let g: Metric<3> = euclidean();

    let s = Vector::<3>::scalar(0.0, g);
    let result = exp(&s);

    assert!((result.scalar_part() - 1.0).abs() < 1e-12);
}

// =============================================================================
// exp: Bivector
// =============================================================================

#[test]
fn exp_zero_bivector() {
    let g: Metric<3> = euclidean();

    let b = Vector::<3>::zero(2, g);
    let result = exp(&b);

    assert!(
        (result.scalar_part() - 1.0).abs() < 1e-12,
        "exp(0) should be 1",
    );
}

#[test]
fn exp_bivector_unit_norm() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let b = wedge(&e[0], &e[1]);
    let result = exp(&b);

    // exp(B) where |B| = 1: cos(1) + B̂ sin(1)
    assert!(
        (result.scalar_part() - 1.0f64.cos()).abs() < 1e-12,
        "scalar part should be cos(1), got {}",
        result.scalar_part(),
    );

    let bv = result.grade_project(2);
    assert!(
        (bv.norm() - 1.0f64.sin()).abs() < 1e-12,
        "bivector norm should be sin(1), got {}",
        bv.norm(),
    );
}

#[test]
fn exp_bivector_is_unit_rotor() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // exp(B) should satisfy R ~R = 1
    let b = &wedge(&e[0], &e[1]) * 0.77;
    let r = exp(&b);
    let product = &r * &r.rev();

    assert!(
        (product.scalar_part() - 1.0).abs() < 1e-12,
        "exp(B) should be a unit rotor: R ~R = {}, not 1",
        product.scalar_part(),
    );
}

#[test]
fn exp_matches_rotor_constructor() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let angle = 1.23;

    // rotor(plane, angle) should equal exp(-B̂ angle/2)
    let r = rotor(&plane, angle);
    let b_unit = plane.normalize().unwrap();
    let generator = &b_unit * (-angle / 2.0);
    let via_exp = exp(&generator);

    assert!(
        (r.scalar_part() - via_exp.scalar_part()).abs() < 1e-12,
        "scalar parts should match",
    );

    let r_bv = r.grade_project(2);
    let exp_bv = via_exp.grade_project(2);
    assert!(
        (&r_bv - &exp_bv).is_zero(1e-12),
        "bivector parts should match",
    );
}

// =============================================================================
// exp: Algebraic Properties
// =============================================================================

#[test]
fn exp_negative_is_inverse() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // exp(-B) = exp(B)^{-1}, so exp(B) * exp(-B) = 1
    let b = &wedge(&e[0], &e[1]) * 0.5;
    let neg_b = -&b;

    let r = exp(&b);
    let r_neg = exp(&neg_b);
    let product = &r * &r_neg;

    assert!(
        (product.scalar_part() - 1.0).abs() < 1e-12,
        "exp(B) * exp(-B) should be 1",
    );
}

#[test]
fn exp_double_is_squared() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // exp(2B) = exp(B) * exp(B) for bivectors in the same plane
    let b = &wedge(&e[0], &e[1]) * 0.3;
    let b2 = &b * 2.0;

    let exp_2b = exp(&b2);
    let exp_b = exp(&b);
    let squared = &exp_b * &exp_b;

    assert!((exp_2b.scalar_part() - squared.scalar_part()).abs() < 1e-12,);

    let diff = exp_2b.grade_project(2) - squared.grade_project(2);
    assert!(diff.is_zero(1e-12), "exp(2B) should equal exp(B)²");
}

// =============================================================================
// log
// =============================================================================

#[test]
fn log_exp_roundtrip() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let b = &wedge(&e[0], &e[1]) * 0.77;
    let r = exp(&b);
    let recovered = log(&r);

    assert_eq!(recovered.grade(), 2);
    assert!(
        (&recovered - &b).is_zero(1e-12),
        "log(exp(B)) should equal B",
    );
}

#[test]
fn exp_log_roundtrip() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, 1.0);
    let b = log(&r);
    let recovered = exp(&b);

    assert!((r.scalar_part() - recovered.scalar_part()).abs() < 1e-12,);

    let diff = r.grade_project(2) - recovered.grade_project(2);
    assert!(diff.is_zero(1e-12), "exp(log(R)) should equal R");
}

#[test]
fn log_identity_is_zero() {
    let g: Metric<3> = euclidean();

    let identity = morphis::multivector::MultiVector::from_vector(Vector::<3>::scalar(1.0, g));
    let b = log(&identity);

    assert_eq!(b.grade(), 2);
    assert!(b.is_zero(1e-12), "log(1) should be zero bivector");
}

#[test]
fn log_extracts_correct_angle() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let angle = 1.5;
    let r = rotor(&plane, angle);

    // log(R) should have norm = angle/2
    let b = log(&r);
    assert!(
        (b.norm() - angle / 2.0).abs() < 1e-12,
        "log(R) norm should be angle/2 = {}, got {}",
        angle / 2.0,
        b.norm(),
    );
}

// =============================================================================
// slerp
// =============================================================================

#[test]
fn slerp_endpoints() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r0 = rotor(&plane, 0.3);
    let r1 = rotor(&plane, 1.5);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));

    // slerp at t=0 should match R0
    let at_0 = slerp(&r0, &r1, 0.0);
    let result_0 = transform(&v, &at_0);
    let expected_0 = transform(&v, &r0);

    for m in 0..3 {
        assert!(
            (result_0.component(&[m]) - expected_0.component(&[m])).abs() < 1e-11,
            "slerp(t=0) should match R0 at component {}",
            m,
        );
    }

    // slerp at t=1 should match R1
    let at_1 = slerp(&r0, &r1, 1.0);
    let result_1 = transform(&v, &at_1);
    let expected_1 = transform(&v, &r1);

    for m in 0..3 {
        assert!(
            (result_1.component(&[m]) - expected_1.component(&[m])).abs() < 1e-11,
            "slerp(t=1) should match R1 at component {}",
            m,
        );
    }
}

#[test]
fn slerp_midpoint() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r0 = rotor(&plane, 0.0);
    let r1 = rotor(&plane, PI / 2.0);

    // Midpoint should be rotation by π/4
    let mid = slerp(&r0, &r1, 0.5);
    let expected = rotor(&plane, PI / 4.0);

    let v = &e[0];
    let result = transform(v, &mid);
    let expected_result = transform(v, &expected);

    for m in 0..3 {
        assert!(
            (result.component(&[m]) - expected_result.component(&[m])).abs() < 1e-11,
            "slerp midpoint should match π/4 rotation at component {}",
            m,
        );
    }
}

#[test]
fn slerp_result_is_unit_rotor() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let r0 = rotor(&wedge(&e[0], &e[1]), 0.3);
    let r1 = rotor(&wedge(&e[0], &e[1]), 1.5);

    // Interpolated rotor should be unit: R ~R = 1
    let mid = slerp(&r0, &r1, 0.5);
    let product = &mid * &mid.rev();

    assert!(
        (product.scalar_part() - 1.0).abs() < 1e-11,
        "slerp result should be unit rotor, got R ~R = {}",
        product.scalar_part(),
    );
}

#[test]
fn slerp_different_planes() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let r0 = rotor(&wedge(&e[0], &e[1]), 0.5);
    let r1 = rotor(&wedge(&e[1], &e[2]), 0.5);

    // Should still produce valid unit rotors at intermediate points
    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let r = slerp(&r0, &r1, t);
        let product = &r * &r.rev();
        assert!(
            (product.scalar_part() - 1.0).abs() < 1e-10,
            "slerp at t={} should be unit rotor, got R ~R = {}",
            t,
            product.scalar_part(),
        );
    }
}

// =============================================================================
// Regularized Solve
// =============================================================================

#[test]
fn regularized_solve_converges_to_unregularized() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let mat = DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ]);
    let l = Operator::from_matrix(&mat, 1, 1, g);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));
    let y = l.apply(&v);

    // With very small alpha, should approximate unregularized solve
    let unreg = l.solve(&y);
    let reg = l.solve_regularized(&y, 1e-14);

    for m in 0..3 {
        assert!(
            (unreg.component(&[m]) - reg.component(&[m])).abs() < 1e-8,
            "regularized solve with tiny alpha should match unregularized at component {}",
            m,
        );
    }
}

#[test]
fn regularized_solve_shrinks_solution() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let mat = DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ]);
    let l = Operator::from_matrix(&mat, 1, 1, g);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));
    let y = l.apply(&v);

    // Larger alpha should produce smaller solution norm
    let small_alpha = l.solve_regularized(&y, 0.01);
    let large_alpha = l.solve_regularized(&y, 10.0);

    assert!(
        large_alpha.norm() < small_alpha.norm(),
        "larger alpha should produce smaller solution: {} vs {}",
        large_alpha.norm(),
        small_alpha.norm(),
    );
}

// =============================================================================
// Lorentzian (Hyperbolic) exp/log
// =============================================================================

#[test]
fn exp_timelike_bivector() {
    let g: Metric<4> = lorentzian();
    let e = basis(g);

    // e0 ^ e1 is a timelike plane: e0² = +1, e1² = -1
    // B² = -(e0²)(e1²) = -1 * (-1) = +1 > 0 → hyperbolic
    let b = &wedge(&e[0], &e[1]) * 0.5;
    let result = exp(&b);

    // Should use cosh/sinh: exp(B) = cosh(θ) + B̂ sinh(θ)
    // norm_squared of this bivector is negative (timelike)
    let ns = b.norm_squared();
    assert!(
        ns < 0.0,
        "timelike bivector should have negative norm_squared, got {}",
        ns
    );

    let theta = (-ns).sqrt();
    assert!(
        (result.scalar_part() - theta.cosh()).abs() < 1e-12,
        "scalar part should be cosh(θ) = {}, got {}",
        theta.cosh(),
        result.scalar_part(),
    );

    // Result should NOT be a unit rotor in the circular sense (scalar > 1)
    assert!(
        result.scalar_part() > 1.0,
        "hyperbolic exp scalar should exceed 1, got {}",
        result.scalar_part(),
    );
}

#[test]
fn exp_log_roundtrip_lorentzian() {
    let g: Metric<4> = lorentzian();
    let e = basis(g);

    // Timelike bivector
    let b = &wedge(&e[0], &e[1]) * 0.7;
    let r = exp(&b);
    let recovered = log(&r);

    assert_eq!(recovered.grade(), 2);
    assert!(
        (&recovered - &b).is_zero(1e-12),
        "log(exp(B)) should equal B for timelike bivector",
    );
}

#[test]
fn log_exp_roundtrip_lorentzian() {
    let g: Metric<4> = lorentzian();
    let e = basis(g);

    let b = &wedge(&e[0], &e[1]) * 0.4;
    let r = exp(&b);
    let b_recovered = log(&r);
    let r_recovered = exp(&b_recovered);

    assert!((r.scalar_part() - r_recovered.scalar_part()).abs() < 1e-12,);

    let diff = r.grade_project(2) - r_recovered.grade_project(2);
    assert!(
        diff.is_zero(1e-12),
        "exp(log(R)) should equal R for hyperbolic rotor"
    );
}

#[test]
fn exp_negative_timelike_is_inverse() {
    let g: Metric<4> = lorentzian();
    let e = basis(g);

    // exp(B) * exp(-B) = 1 for timelike bivectors too
    let b = &wedge(&e[0], &e[1]) * 0.6;
    let neg_b = -&b;

    let r = exp(&b);
    let r_neg = exp(&neg_b);
    let product = &r * &r_neg;

    assert!(
        (product.scalar_part() - 1.0).abs() < 1e-12,
        "exp(B) * exp(-B) should be 1 for timelike bivector, got scalar {}",
        product.scalar_part(),
    );
}

#[test]
fn exp_spacelike_in_lorentzian() {
    let g: Metric<4> = lorentzian();
    let e = basis(g);

    // e1 ^ e2 is a spacelike plane in Lorentzian: e1² = -1, e2² = -1
    // B² = -(e1²)(e2²) = -(-1)(-1) = -1 < 0 → circular (same as Euclidean)
    let b = &wedge(&e[1], &e[2]) * 0.5;
    let ns = b.norm_squared();
    assert!(
        ns > 0.0,
        "spacelike bivector should have positive norm_squared, got {}",
        ns
    );

    let result = exp(&b);

    // Should be a circular rotor: scalar in [-1, 1]
    assert!(
        result.scalar_part().abs() <= 1.0 + 1e-12,
        "spacelike exp in Lorentzian should be circular, scalar = {}",
        result.scalar_part(),
    );
}
