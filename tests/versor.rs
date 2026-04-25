use std::f64::consts::PI;

use morphis::metric::{Metric, euclidean, lorentzian};
use morphis::ops::wedge;
use morphis::vector::basis;
use morphis::versor::rotor;

// =============================================================================
// Rotor Construction
// =============================================================================

#[test]
fn rotor_zero_angle_is_identity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, 0.0);

    // Motor should be scalar 1 + bivector 0
    let motor = r.motor();
    assert!((motor.scalar_part() - 1.0).abs() < 1e-12);
}

#[test]
fn rotor_is_even_grade() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, PI / 4.0);

    assert!(r.is_even());
}

#[test]
fn rotor_motor_is_unit_norm() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, 1.23);

    // R ~R should be scalar 1
    let motor = r.motor();
    let product = motor.clone() * motor.rev();
    assert!(
        (product.scalar_part() - 1.0).abs() < 1e-12,
        "rotor motor should have unit norm, got scalar part {}",
        product.scalar_part(),
    );
}

// =============================================================================
// Rotation Tests
// =============================================================================

#[test]
fn rotate_e1_by_90_in_e12_plane() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, PI / 2.0);

    // Rotating e1 by 90° in the e1-e2 plane should give e2
    let rotated = r.transform(&e[0]);

    assert_eq!(rotated.grade(), 1);
    assert!(
        rotated.component(&[0]).abs() < 1e-12,
        "e1 component should vanish, got {}",
        rotated.component(&[0]),
    );
    assert!(
        (rotated.component(&[1]) - 1.0).abs() < 1e-12,
        "e2 component should be 1, got {}",
        rotated.component(&[1]),
    );
    assert!(
        rotated.component(&[2]).abs() < 1e-12,
        "e3 component should vanish, got {}",
        rotated.component(&[2]),
    );
}

#[test]
fn rotate_e2_by_90_in_e12_plane() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, PI / 2.0);

    // Rotating e2 by 90° in the e1-e2 plane should give -e1
    let rotated = r.transform(&e[1]);

    assert!(
        (rotated.component(&[0]) + 1.0).abs() < 1e-12,
        "e1 component should be -1, got {}",
        rotated.component(&[0]),
    );
    assert!(
        rotated.component(&[1]).abs() < 1e-12,
        "e2 component should vanish, got {}",
        rotated.component(&[1]),
    );
}

#[test]
fn rotate_preserves_orthogonal_component() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, PI / 3.0);

    // Rotating e3 in the e1-e2 plane should leave it unchanged
    let rotated = r.transform(&e[2]);

    assert!(rotated.component(&[0]).abs() < 1e-12);
    assert!(rotated.component(&[1]).abs() < 1e-12);
    assert!((rotated.component(&[2]) - 1.0).abs() < 1e-12);
}

#[test]
fn rotate_preserves_norm() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &(&e[0] * 3.0) + &(&e[1] * 4.0);
    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, 0.77);

    let rotated = r.transform(&v);

    assert!(
        (rotated.norm() - v.norm()).abs() < 1e-12,
        "rotation should preserve norm: {} vs {}",
        rotated.norm(),
        v.norm(),
    );
}

#[test]
fn rotate_by_360_is_identity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));
    let plane = wedge(&e[0], &e[2]);
    let r = rotor(&plane, 2.0 * PI);

    let rotated = r.transform(&v);

    for m in 0..3 {
        assert!(
            (rotated.component(&[m]) - v.component(&[m])).abs() < 1e-11,
            "360° rotation should be identity at component {}",
            m,
        );
    }
}

#[test]
fn rotate_general_vector_in_3d() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // Rotate (1, 0, 0) by 180° in e1-e2 plane -> (-1, 0, 0)
    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, PI);

    let rotated = r.transform(&e[0]);

    assert!(
        (rotated.component(&[0]) + 1.0).abs() < 1e-11,
        "180° rotation of e1 in e12 plane should give -e1, got {}",
        rotated.component(&[0]),
    );
    assert!(rotated.component(&[1]).abs() < 1e-11);
    assert!(rotated.component(&[2]).abs() < 1e-11);
}

// =============================================================================
// Bivector Rotation
// =============================================================================

#[test]
fn rotate_bivector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // Rotate the e1^e3 bivector by 90° in the e1-e2 plane
    // e1 -> e2, e3 -> e3, so e1^e3 -> e2^e3
    let bv = wedge(&e[0], &e[2]);
    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, PI / 2.0);

    let rotated = r.transform(&bv);

    assert_eq!(rotated.grade(), 2);
    // e2^e3 component should be 1
    assert!(
        (rotated.component(&[1, 2]) - 1.0).abs() < 1e-12,
        "e2^e3 component should be 1, got {}",
        rotated.component(&[1, 2]),
    );
    // e1^e3 component should vanish
    assert!(
        rotated.component(&[0, 2]).abs() < 1e-12,
        "e1^e3 component should vanish, got {}",
        rotated.component(&[0, 2]),
    );
}

// =============================================================================
// Operator Syntax
// =============================================================================

#[test]
fn sandwich_product_operator() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, PI / 2.0);

    // R % e1 should equal R.transform(e1)
    let via_method = r.transform(&e[0]);
    let via_operator = &r % &e[0];

    for m in 0..3 {
        assert!(
            (via_method.component(&[m]) - via_operator.component(&[m])).abs() < 1e-12,
            "% operator should match transform() at component {}",
            m,
        );
    }
}

#[test]
fn composition_operator() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r45 = rotor(&plane, PI / 4.0);

    // Two 45° rotations = one 90° rotation
    let r90 = &r45 * &r45;
    let direct = rotor(&plane, PI / 2.0);

    let v = &(&e[0] * 2.0) + &(&e[1] * 3.0);
    let composed = r90.transform(&v);
    let expected = direct.transform(&v);

    for m in 0..3 {
        assert!(
            (composed.component(&[m]) - expected.component(&[m])).abs() < 1e-11,
            "composed rotation should match direct at component {}",
            m,
        );
    }
}

#[test]
fn inverse_operator() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, 0.77);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));

    // !R % (R % v) should recover v
    let rotated = &r % &v;
    let recovered = &(!&r) % &rotated;

    for m in 0..3 {
        assert!(
            (recovered.component(&[m]) - v.component(&[m])).abs() < 1e-11,
            "inverse should recover original at component {}",
            m,
        );
    }
}

// =============================================================================
// Composition Identities
// =============================================================================

#[test]
fn composition_associativity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let r1 = rotor(&wedge(&e[0], &e[1]), 0.3);
    let r2 = rotor(&wedge(&e[1], &e[2]), 0.5);
    let r3 = rotor(&wedge(&e[0], &e[2]), 0.7);

    let v = &(&e[0] * 1.0) + &(&(&e[1] * 2.0) + &(&e[2] * 3.0));

    // (R3 * R2) * R1 should equal R3 * (R2 * R1) when acting on v
    let left = (&(&r3 * &r2) * &r1).transform(&v);
    let right = (&r3 * &(&r2 * &r1)).transform(&v);

    for m in 0..3 {
        assert!(
            (left.component(&[m]) - right.component(&[m])).abs() < 1e-10,
            "composition should be associative at component {}",
            m,
        );
    }
}

#[test]
fn inverse_composition_identity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let r = rotor(&wedge(&e[0], &e[1]), 1.23);
    let v = &(&e[0] * 2.0) + &(&e[1] * 3.0);

    // R * R^{-1} should act as identity
    let product = &r * &r.inv();
    let result = product.transform(&v);

    for m in 0..3 {
        assert!(
            (result.component(&[m]) - v.component(&[m])).abs() < 1e-11,
            "R * R^-1 should be identity at component {}",
            m,
        );
    }
}

// =============================================================================
// Higher Dimension
// =============================================================================

#[test]
fn rotor_in_4d() {
    let g: Metric<4> = euclidean();
    let e = basis(g);

    // Rotate in the e1-e3 plane by 90°
    let plane = wedge(&e[0], &e[2]);
    let r = rotor(&plane, PI / 2.0);

    // e1 -> e3
    let rotated = r.transform(&e[0]);
    assert!(rotated.component(&[0]).abs() < 1e-12);
    assert!(rotated.component(&[1]).abs() < 1e-12);
    assert!((rotated.component(&[2]) - 1.0).abs() < 1e-12);
    assert!(rotated.component(&[3]).abs() < 1e-12);

    // e2 and e4 unchanged
    let rotated_e2 = r.transform(&e[1]);
    assert!((rotated_e2.component(&[1]) - 1.0).abs() < 1e-12);

    let rotated_e4 = r.transform(&e[3]);
    assert!((rotated_e4.component(&[3]) - 1.0).abs() < 1e-12);
}

// =============================================================================
// Lorentzian Signature
// =============================================================================

#[test]
fn rotor_lorentzian_spacelike_plane() {
    let g: Metric<4> = lorentzian();
    let e = basis(g);

    // Rotation in a spacelike plane (e2-e3). In Lorentzian signature,
    // the metric contraction reverses the rotation sense relative to
    // Euclidean: R % e2 -> -e3 (not +e3).
    let plane = wedge(&e[1], &e[2]);
    let r = rotor(&plane, PI / 2.0);

    let rotated = r.transform(&e[1]);
    assert!(rotated.component(&[0]).abs() < 1e-12);
    assert!(rotated.component(&[1]).abs() < 1e-12);
    assert!((rotated.component(&[2]) + 1.0).abs() < 1e-11);
    assert!(rotated.component(&[3]).abs() < 1e-12);

    // Norm preserved (spacelike norm)
    assert!((rotated.norm() - e[1].norm()).abs() < 1e-12);
}
