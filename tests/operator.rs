use nalgebra::DMatrix;
use ndarray::{ArrayD, IxDyn};

use morphis::metric::{Metric, euclidean};
use morphis::operator::Operator;
use morphis::ops::{interior_left, wedge};
use morphis::vector::basis;

// =============================================================================
// Identity
// =============================================================================

#[test]
fn identity_grade_1() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // Identity on grade-1: D×D identity matrix reshaped to [D, D] tensor
    let id_matrix = DMatrix::identity(3, 3);
    let id = Operator::from_matrix(&id_matrix, 1, 1, g);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));
    let result = id.apply(&v);

    for m in 0..3 {
        assert!(
            (result.component(&[m]) - v.component(&[m])).abs() < 1e-12,
            "identity failed at component {}",
            m,
        );
    }
}

#[test]
fn identity_grade_2() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // Identity on grade-2: D²×D² identity
    let id_matrix = DMatrix::identity(9, 9);
    let id = Operator::from_matrix(&id_matrix, 2, 2, g);

    let bv = wedge(&e[0], &e[1]);
    let result = id.apply(&bv);

    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (result.component(&[m, n]) - bv.component(&[m, n])).abs() < 1e-12,
                "identity on bivector failed at [{}, {}]",
                m,
                n,
            );
        }
    }
}

// =============================================================================
// Linearity
// =============================================================================

#[test]
fn linearity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // Random 3×3 operator on grade-1 vectors
    #[rustfmt::skip]
    let matrix = DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ]);
    let l = Operator::from_matrix(&matrix, 1, 1, g);

    let u = &(&e[0] * 2.0) + &(&e[1] * 3.0);
    let v = &(&e[1] * 1.0) + &(&e[2] * 4.0);
    let alpha = 2.5;
    let beta = -1.3;

    let lhs = l.apply(&(&(&u * alpha) + &(&v * beta)));
    let rhs = &(&l.apply(&u) * alpha) + &(&l.apply(&v) * beta);

    for m in 0..3 {
        assert!(
            (lhs.component(&[m]) - rhs.component(&[m])).abs() < 1e-12,
            "linearity failed at component {}",
            m,
        );
    }
}

// =============================================================================
// Composition
// =============================================================================

#[test]
fn composition() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let l_mat = DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ]);
    #[rustfmt::skip]
    let m_mat = DMatrix::from_row_slice(3, 3, &[
        1.0, 0.0, 1.0,
        2.0, 1.0, 0.0,
        0.0, 1.0, 1.0,
    ]);
    let l = Operator::from_matrix(&l_mat, 1, 1, g);
    let m = Operator::from_matrix(&m_mat, 1, 1, g);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 1.0));

    // (L * M)(v) = L(M(v))
    let composed = &l * &m;
    let lhs = composed.apply(&v);
    let rhs = l.apply(&m.apply(&v));

    for m in 0..3 {
        assert!(
            (lhs.component(&[m]) - rhs.component(&[m])).abs() < 1e-12,
            "composition failed at component {}",
            m,
        );
    }
}

#[test]
fn mul_operator_syntax() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let mat = DMatrix::from_row_slice(3, 3, &[
        2.0, 0.0, 0.0,
        0.0, 3.0, 0.0,
        0.0, 0.0, 4.0,
    ]);
    let l = Operator::from_matrix(&mat, 1, 1, g);

    // L * v via operator
    let result = &l * &e[0];
    assert!((result.component(&[0]) - 2.0).abs() < 1e-12);
}

// =============================================================================
// Adjoint
// =============================================================================

#[test]
fn adjoint_inner_product_identity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let mat = DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ]);
    let l = Operator::from_matrix(&mat, 1, 1, g);
    let l_adj = l.adjoint();

    let u = &(&e[0] * 2.0) + &(&e[1] * 3.0);
    let v = &(&e[0] * 1.0) + &(&(&e[1] * 4.0) + &(&e[2] * 2.0));

    // <L(u), v> = <u, L†(v)>
    let lu = l.apply(&u);
    let lav = l_adj.apply(&v);

    let lhs = interior_left(&lu, &v);
    let rhs = interior_left(&u, &lav);

    assert!(
        (lhs.data[IxDyn(&[])] - rhs.data[IxDyn(&[])]).abs() < 1e-12,
        "<L(u), v> = {} but <u, L†(v)> = {}",
        lhs.data[IxDyn(&[])],
        rhs.data[IxDyn(&[])],
    );
}

#[test]
fn adjoint_of_adjoint_is_original() {
    let g: Metric<3> = euclidean();

    #[rustfmt::skip]
    let mat = DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ]);
    let l = Operator::from_matrix(&mat, 1, 1, g);
    let l_aa = l.adjoint().adjoint();

    let original = l.to_matrix();
    let roundtrip = l_aa.to_matrix();

    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (original[(m, n)] - roundtrip[(m, n)]).abs() < 1e-12,
                "(L†)† should equal L at ({}, {})",
                m,
                n,
            );
        }
    }
}

// =============================================================================
// SVD
// =============================================================================

#[test]
fn svd_roundtrip() {
    let g: Metric<3> = euclidean();

    #[rustfmt::skip]
    let mat = DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ]);
    let l = Operator::from_matrix(&mat, 1, 1, g);

    let (u, sigma, vt) = l.svd();

    // Reconstruct: U * diag(σ) * V†
    let mut sigma_mat = DMatrix::zeros(u.ncols(), vt.nrows());
    for (m, &s) in sigma.iter().enumerate() {
        sigma_mat[(m, m)] = s;
    }
    let reconstructed = &u * sigma_mat * &vt;

    let original = l.to_matrix();
    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (original[(m, n)] - reconstructed[(m, n)]).abs() < 1e-12,
                "SVD roundtrip failed at ({}, {}): {} vs {}",
                m,
                n,
                original[(m, n)],
                reconstructed[(m, n)],
            );
        }
    }
}

#[test]
fn svd_singular_values_descending() {
    let g: Metric<3> = euclidean();

    #[rustfmt::skip]
    let mat = DMatrix::from_row_slice(3, 3, &[
        5.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 0.5,
    ]);
    let l = Operator::from_matrix(&mat, 1, 1, g);

    let (_, sigma, _) = l.svd();

    for m in 1..sigma.len() {
        assert!(
            sigma[m - 1] >= sigma[m],
            "singular values should be descending: σ_{} = {} < σ_{} = {}",
            m - 1,
            sigma[m - 1],
            m,
            sigma[m],
        );
    }
}

// =============================================================================
// Pseudoinverse
// =============================================================================

#[test]
fn pseudoinverse_property() {
    let g: Metric<3> = euclidean();

    #[rustfmt::skip]
    let mat = DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ]);
    let l = Operator::from_matrix(&mat, 1, 1, g);

    // L * L⁺ * L = L
    let l_pinv = l.pseudoinverse();
    let roundtrip = &(&l * &l_pinv) * &l;

    let original = l.to_matrix();
    let result = roundtrip.to_matrix();

    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (original[(m, n)] - result[(m, n)]).abs() < 1e-11,
                "L L⁺ L should equal L at ({}, {}): {} vs {}",
                m,
                n,
                original[(m, n)],
                result[(m, n)],
            );
        }
    }
}

#[test]
fn pseudoinverse_of_invertible_is_inverse() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    #[rustfmt::skip]
    let mat = DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ]);
    let l = Operator::from_matrix(&mat, 1, 1, g);
    let l_pinv = l.pseudoinverse();

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));

    // L⁺(L(v)) ≈ v for full-rank L
    let roundtrip = l_pinv.apply(&l.apply(&v));

    for m in 0..3 {
        assert!(
            (roundtrip.component(&[m]) - v.component(&[m])).abs() < 1e-11,
            "pseudoinverse roundtrip failed at component {}",
            m,
        );
    }
}

// =============================================================================
// Solve
// =============================================================================

#[test]
fn solve_recovers_input() {
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

    // solve(L, y) should recover v for full-rank L
    let recovered = l.solve(&y);

    for m in 0..3 {
        assert!(
            (recovered.component(&[m]) - v.component(&[m])).abs() < 1e-11,
            "solve failed at component {}: {} vs {}",
            m,
            recovered.component(&[m]),
            v.component(&[m]),
        );
    }
}

// =============================================================================
// Cross-Grade Operators
// =============================================================================

#[test]
fn grade_1_to_grade_2_operator() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // Build a simple grade-1 → grade-2 operator
    // Maps e1 → e12, e2 → e23, e3 → e13
    let mut data = ArrayD::zeros(IxDyn(&[3, 3, 3]));
    // e1 → e12: data[0, 1, 0] = 1, data[1, 0, 0] = -1
    data[IxDyn(&[0, 1, 0])] = 1.0;
    data[IxDyn(&[1, 0, 0])] = -1.0;
    // e2 → e23: data[1, 2, 1] = 1, data[2, 1, 1] = -1
    data[IxDyn(&[1, 2, 1])] = 1.0;
    data[IxDyn(&[2, 1, 1])] = -1.0;
    // e3 → e13: data[0, 2, 2] = 1, data[2, 0, 2] = -1
    data[IxDyn(&[0, 2, 2])] = 1.0;
    data[IxDyn(&[2, 0, 2])] = -1.0;

    let l = Operator::new(data, 1, 2, g);

    let result = l.apply(&e[0]);
    assert_eq!(result.grade(), 2);
    assert!((result.component(&[0, 1]) - 1.0).abs() < 1e-12);
    assert!((result.component(&[1, 0]) + 1.0).abs() < 1e-12);
}

// =============================================================================
// Matrix Roundtrip
// =============================================================================

#[test]
fn to_matrix_from_matrix_roundtrip() {
    let g: Metric<3> = euclidean();

    #[rustfmt::skip]
    let mat = DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, 0.0,
        0.0, 3.0, 1.0,
        1.0, 0.0, 2.0,
    ]);
    let l = Operator::from_matrix(&mat, 1, 1, g);
    let roundtrip = l.to_matrix();

    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (mat[(m, n)] - roundtrip[(m, n)]).abs() < 1e-12,
                "matrix roundtrip failed at ({}, {})",
                m,
                n,
            );
        }
    }
}
