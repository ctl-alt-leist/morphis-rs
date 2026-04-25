use std::f64::consts::PI;

use morphis::metric::{Metric, euclidean, lorentzian};
use morphis::ops::wedge;
use morphis::vector::basis;
use morphis::versor::{rotor, transform};

// =============================================================================
// Rotor Construction
// =============================================================================

#[test]
fn rotor_zero_angle_is_identity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, 0.0);

    assert!((r.scalar_part() - 1.0).abs() < 1e-12);
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
fn rotor_is_unit_norm() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, 1.23);

    // R ~R should be scalar 1
    let product = &r * &r.rev();
    assert!(
        (product.scalar_part() - 1.0).abs() < 1e-12,
        "R ~R should be 1, got scalar part {}",
        product.scalar_part(),
    );
}

// =============================================================================
// Rotation via transform()
// =============================================================================

#[test]
fn rotate_e1_by_90_in_e12_plane() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, PI / 2.0);

    let rotated = transform(&e[0], &r);

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

    let rotated = transform(&e[1], &r);

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

    let rotated = transform(&e[2], &r);

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

    let rotated = transform(&v, &r);

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

    let rotated = transform(&v, &r);

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

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, PI);

    let rotated = transform(&e[0], &r);

    assert!(
        (rotated.component(&[0]) + 1.0).abs() < 1e-11,
        "180° rotation of e1 in e12 plane should give -e1, got {}",
        rotated.component(&[0]),
    );
    assert!(rotated.component(&[1]).abs() < 1e-11);
    assert!(rotated.component(&[2]).abs() < 1e-11);
}

// =============================================================================
// Explicit Sandwich Product
// =============================================================================

#[test]
fn explicit_sandwich_matches_transform() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, PI / 2.0);

    // Explicit: R * v * ~R, then grade-project
    let explicit = (&(&r * &e[0]) * &r.rev()).grade_project(1);
    let via_transform = transform(&e[0], &r);

    for m in 0..3 {
        assert!(
            (explicit.component(&[m]) - via_transform.component(&[m])).abs() < 1e-12,
            "explicit sandwich should match transform at component {}",
            m,
        );
    }
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

    let rotated = transform(&bv, &r);

    assert_eq!(rotated.grade(), 2);
    assert!(
        (rotated.component(&[1, 2]) - 1.0).abs() < 1e-12,
        "e2^e3 component should be 1, got {}",
        rotated.component(&[1, 2]),
    );
    assert!(
        rotated.component(&[0, 2]).abs() < 1e-12,
        "e1^e3 component should vanish, got {}",
        rotated.component(&[0, 2]),
    );
}

// =============================================================================
// Composition and Inverse
// =============================================================================

#[test]
fn composition_via_geometric_product() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r45 = rotor(&plane, PI / 4.0);

    // Two 45° rotations = one 90° rotation
    // Composition is just the geometric product of the rotors
    let r90 = &r45 * &r45;
    let direct = rotor(&plane, PI / 2.0);

    let v = &(&e[0] * 2.0) + &(&e[1] * 3.0);
    let composed = transform(&v, &r90);
    let expected = transform(&v, &direct);

    for m in 0..3 {
        assert!(
            (composed.component(&[m]) - expected.component(&[m])).abs() < 1e-11,
            "composed rotation should match direct at component {}",
            m,
        );
    }
}

#[test]
fn inverse_via_reverse() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let plane = wedge(&e[0], &e[1]);
    let r = rotor(&plane, 0.77);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));

    // R^{-1} = ~R for a unit rotor. Applying ~R undoes R.
    let rotated = transform(&v, &r);
    let r_rev = r.rev();
    let recovered = transform(&rotated, &r_rev);

    for m in 0..3 {
        assert!(
            (recovered.component(&[m]) - v.component(&[m])).abs() < 1e-11,
            "reverse should undo rotation at component {}",
            m,
        );
    }
}

#[test]
fn composition_associativity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let r1 = rotor(&wedge(&e[0], &e[1]), 0.3);
    let r2 = rotor(&wedge(&e[1], &e[2]), 0.5);
    let r3 = rotor(&wedge(&e[0], &e[2]), 0.7);

    let v = &(&e[0] * 1.0) + &(&(&e[1] * 2.0) + &(&e[2] * 3.0));

    // (R3 * R2) * R1 should equal R3 * (R2 * R1)
    let left = transform(&v, &(&(&r3 * &r2) * &r1));
    let right = transform(&v, &(&r3 * &(&r2 * &r1)));

    for m in 0..3 {
        assert!(
            (left.component(&[m]) - right.component(&[m])).abs() < 1e-10,
            "composition should be associative at component {}",
            m,
        );
    }
}

#[test]
fn r_times_r_inverse_is_identity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let r = rotor(&wedge(&e[0], &e[1]), 1.23);
    let v = &(&e[0] * 2.0) + &(&e[1] * 3.0);

    // R * ~R should act as identity on v
    let r_identity = &r * &r.rev();
    let result = transform(&v, &r_identity);

    for m in 0..3 {
        assert!(
            (result.component(&[m]) - v.component(&[m])).abs() < 1e-11,
            "R * ~R should be identity at component {}",
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

    let plane = wedge(&e[0], &e[2]);
    let r = rotor(&plane, PI / 2.0);

    // e1 -> e3
    let rotated = transform(&e[0], &r);
    assert!(rotated.component(&[0]).abs() < 1e-12);
    assert!(rotated.component(&[1]).abs() < 1e-12);
    assert!((rotated.component(&[2]) - 1.0).abs() < 1e-12);
    assert!(rotated.component(&[3]).abs() < 1e-12);

    // e2 and e4 unchanged
    let rotated_e2 = transform(&e[1], &r);
    assert!((rotated_e2.component(&[1]) - 1.0).abs() < 1e-12);

    let rotated_e4 = transform(&e[3], &r);
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
    // Euclidean: transform(e2, R) -> -e3 (not +e3).
    let plane = wedge(&e[1], &e[2]);
    let r = rotor(&plane, PI / 2.0);

    let rotated = transform(&e[1], &r);
    assert!(rotated.component(&[0]).abs() < 1e-12);
    assert!(rotated.component(&[1]).abs() < 1e-12);
    assert!((rotated.component(&[2]) + 1.0).abs() < 1e-11);
    assert!(rotated.component(&[3]).abs() < 1e-12);

    // Norm preserved (spacelike norm)
    assert!((rotated.norm() - e[1].norm()).abs() < 1e-12);
}
