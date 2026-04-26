use morphis::metric::{Metric, euclidean};
use morphis::multivector::MultiVector;
use morphis::ops::wedge;
use morphis::outermorphism::Outermorphism;
use morphis::vector::{basis, pseudoscalar};

// =============================================================================
// Identity
// =============================================================================

#[test]
fn identity_on_grade_1() {
    let g: Metric<3> = euclidean();
    let e = basis(g);
    let id = Outermorphism::identity(g);

    for m in 0..3 {
        let result = id.apply(&e[m]);
        for n in 0..3 {
            let expected = if m == n { 1.0 } else { 0.0 };
            assert!(
                (result.component(&[n]) - expected).abs() < 1e-12,
                "identity failed on e{} at component {}",
                m + 1,
                n,
            );
        }
    }
}

#[test]
fn identity_on_grade_2() {
    let g: Metric<3> = euclidean();
    let e = basis(g);
    let id = Outermorphism::identity(g);

    let b = wedge(&e[0], &e[1]);
    let result = id.apply(&b);

    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (result.component(&[m, n]) - b.component(&[m, n])).abs() < 1e-12,
                "identity on bivector failed at [{}, {}]",
                m,
                n,
            );
        }
    }
}

#[test]
fn identity_on_pseudoscalar() {
    let g: Metric<3> = euclidean();
    let id = Outermorphism::identity(g);

    let ps = pseudoscalar(g);
    let result = id.apply(&ps);

    assert!((result.component(&[0, 1, 2]) - ps.component(&[0, 1, 2])).abs() < 1e-12,);
}

// =============================================================================
// Linearity
// =============================================================================

#[test]
fn linearity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let a = Outermorphism::from_row_slice(&[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ], g);

    let u = &(&e[0] * 2.0) + &(&e[1] * 3.0);
    let v = &(&e[1] * 1.0) + &(&e[2] * 4.0);
    let alpha = 2.5;
    let beta = -1.3;

    // A(α u + β v) = α A(u) + β A(v)
    let lhs = a.apply(&(&(&u * alpha) + &(&v * beta)));
    let rhs = &(&a.apply(&u) * alpha) + &(&a.apply(&v) * beta);

    for m in 0..3 {
        assert!(
            (lhs.component(&[m]) - rhs.component(&[m])).abs() < 1e-12,
            "linearity failed at component {}",
            m,
        );
    }
}

// =============================================================================
// Wedge Compatibility (Defining Property)
// =============================================================================

#[test]
fn wedge_compatibility() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let a = Outermorphism::from_row_slice(&[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ], g);

    let u = &(&e[0] * 2.0) + &(&e[1] * 1.0);
    let v = &(&e[1] * 3.0) + &(&e[2] * 1.0);

    // A(u ∧ v) = A(u) ∧ A(v)
    let lhs = a.apply(&wedge(&u, &v));
    let rhs = wedge(&a.apply(&u), &a.apply(&v));

    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (lhs.component(&[m, n]) - rhs.component(&[m, n])).abs() < 1e-12,
                "wedge compatibility failed at [{}, {}]",
                m,
                n,
            );
        }
    }
}

#[test]
fn wedge_compatibility_three_vectors() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let a = Outermorphism::from_row_slice(&[
        1.0, 2.0, 0.0,
        0.0, 1.0, 3.0,
        2.0, 0.0, 1.0,
    ], g);

    // A(e1 ∧ e2 ∧ e3) = A(e1) ∧ A(e2) ∧ A(e3)
    let trivec = wedge(&wedge(&e[0], &e[1]), &e[2]);
    let lhs = a.apply(&trivec);
    let rhs = wedge(&wedge(&a.apply(&e[0]), &a.apply(&e[1])), &a.apply(&e[2]));

    assert!(
        (lhs.component(&[0, 1, 2]) - rhs.component(&[0, 1, 2])).abs() < 1e-12,
        "wedge compatibility on trivector failed",
    );
}

// =============================================================================
// Determinant on Pseudoscalar
// =============================================================================

#[test]
fn determinant_pseudoscalar() {
    let g: Metric<3> = euclidean();

    #[rustfmt::skip]
    let a = Outermorphism::from_row_slice(&[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ], g);

    let ps = pseudoscalar(g);
    let result = a.apply(&ps);

    // A(pseudoscalar) = det(A) * pseudoscalar
    let expected = a.det() * ps.component(&[0, 1, 2]);
    assert!(
        (result.component(&[0, 1, 2]) - expected).abs() < 1e-12,
        "A(I) should be det(A) * I: got {} expected {}",
        result.component(&[0, 1, 2]),
        expected,
    );
}

// =============================================================================
// Composition
// =============================================================================

#[test]
fn composition() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let a = Outermorphism::from_row_slice(&[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ], g);

    #[rustfmt::skip]
    let b = Outermorphism::from_row_slice(&[
        1.0, 0.0, 1.0,
        2.0, 1.0, 0.0,
        0.0, 1.0, 1.0,
    ], g);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 1.0));

    // (A * B)(v) = A(B(v))
    let composed = &a * &b;
    let lhs = composed.apply(&v);
    let rhs = a.apply(&b.apply(&v));

    for m in 0..3 {
        assert!(
            (lhs.component(&[m]) - rhs.component(&[m])).abs() < 1e-12,
            "composition failed at component {}",
            m,
        );
    }
}

#[test]
fn composition_on_bivector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let a = Outermorphism::from_row_slice(&[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ], g);

    #[rustfmt::skip]
    let b = Outermorphism::from_row_slice(&[
        1.0, 0.0, 1.0,
        2.0, 1.0, 0.0,
        0.0, 1.0, 1.0,
    ], g);

    let bv = wedge(&e[0], &e[2]);

    // (A * B)(bivector) = A(B(bivector))
    let composed = &a * &b;
    let lhs = composed.apply(&bv);
    let rhs = a.apply(&b.apply(&bv));

    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (lhs.component(&[m, n]) - rhs.component(&[m, n])).abs() < 1e-12,
                "composition on bivector failed at [{}, {}]",
                m,
                n,
            );
        }
    }
}

// =============================================================================
// Inverse
// =============================================================================

#[test]
fn inverse() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let a = Outermorphism::from_row_slice(&[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ], g);

    let a_inv = a.inv().expect("matrix should be invertible");
    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));

    // A^{-1}(A(v)) = v
    let roundtrip = a_inv.apply(&a.apply(&v));

    for m in 0..3 {
        assert!(
            (roundtrip.component(&[m]) - v.component(&[m])).abs() < 1e-12,
            "inverse roundtrip failed at component {}",
            m,
        );
    }
}

#[test]
fn inverse_composition_is_identity() {
    let g: Metric<3> = euclidean();

    #[rustfmt::skip]
    let a = Outermorphism::from_row_slice(&[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ], g);

    let a_inv = a.inv().unwrap();
    let product = &a * &a_inv;

    // A * A^{-1} should be identity
    let id = Outermorphism::identity(g);
    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (product.matrix()[(m, n)] - id.matrix()[(m, n)]).abs() < 1e-12,
                "A * A^-1 should be identity at ({}, {})",
                m,
                n,
            );
        }
    }
}

// =============================================================================
// Operator Syntax
// =============================================================================

#[test]
fn mul_operator_apply() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let a = Outermorphism::from_row_slice(&[
        0.0, -1.0, 0.0,
        1.0,  0.0, 0.0,
        0.0,  0.0, 1.0,
    ], g);

    // A * e1 = e2 (90° rotation in e1-e2 plane)
    let result = &a * &e[0];
    assert!((result.component(&[0])).abs() < 1e-12);
    assert!((result.component(&[1]) - 1.0).abs() < 1e-12);
    assert!((result.component(&[2])).abs() < 1e-12);
}

#[test]
fn mul_operator_compose() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let id = Outermorphism::identity(g);

    #[rustfmt::skip]
    let a = Outermorphism::from_row_slice(&[
        2.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 2.0,
    ], g);

    // A * I = A
    let result = &a * &id;
    let v = &(&e[0] * 1.0) + &(&e[1] * 2.0);
    let lhs = result.apply(&v);
    let rhs = a.apply(&v);

    for m in 0..3 {
        assert!((lhs.component(&[m]) - rhs.component(&[m])).abs() < 1e-12,);
    }
}

#[test]
fn mul_operator_multivector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let a = Outermorphism::identity(g);
    let mv = MultiVector::from_vector(e[0].clone()) + MultiVector::from_vector(wedge(&e[0], &e[1]));

    let result = &a * &mv;

    // Identity should preserve everything
    assert!(result.grade_select(1).is_some());
    assert!(result.grade_select(2).is_some());
}

// =============================================================================
// Higher Dimension
// =============================================================================

#[test]
fn outermorphism_4d() {
    let g: Metric<4> = euclidean();
    let e = basis(g);

    // Swap e1 and e3, keep e2 and e4
    #[rustfmt::skip]
    let a = Outermorphism::from_row_slice(&[
        0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ], g);

    let result = a.apply(&e[0]);
    assert!((result.component(&[2]) - 1.0).abs() < 1e-12);

    // On bivector e1^e2 -> e3^e2 = -e2^e3
    let bv = wedge(&e[0], &e[1]);
    let result_bv = a.apply(&bv);
    assert!((result_bv.component(&[1, 2]) + 1.0).abs() < 1e-12);
}

// =============================================================================
// Rotation Consistency with Rotor
// =============================================================================

#[test]
fn rotation_matrix_matches_rotor() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // 90° rotation in e1-e2 plane as a matrix
    #[rustfmt::skip]
    let rot = Outermorphism::from_row_slice(&[
        0.0, -1.0, 0.0,
        1.0,  0.0, 0.0,
        0.0,  0.0, 1.0,
    ], g);

    // Same rotation via rotor
    let plane = wedge(&e[0], &e[1]);
    let r = morphis::versor::rotor(&plane, std::f64::consts::PI / 2.0);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));

    let via_matrix = rot.apply(&v);
    let via_rotor = morphis::versor::transform(&v, &r);

    for m in 0..3 {
        assert!(
            (via_matrix.component(&[m]) - via_rotor.component(&[m])).abs() < 1e-12,
            "rotation matrix should match rotor at component {}",
            m,
        );
    }
}

#[test]
fn rotation_matrix_matches_rotor_on_bivector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let rot = Outermorphism::from_row_slice(&[
        0.0, -1.0, 0.0,
        1.0,  0.0, 0.0,
        0.0,  0.0, 1.0,
    ], g);

    let plane = wedge(&e[0], &e[1]);
    let r = morphis::versor::rotor(&plane, std::f64::consts::PI / 2.0);

    let bv = wedge(&e[0], &e[2]);

    let via_matrix = rot.apply(&bv);
    let via_rotor = morphis::versor::transform(&bv, &r);

    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (via_matrix.component(&[m, n]) - via_rotor.component(&[m, n])).abs() < 1e-12,
                "rotation on bivector: matrix vs rotor mismatch at [{}, {}]",
                m,
                n,
            );
        }
    }
}
