use ndarray::IxDyn;

use morphis::metric::{Metric, euclidean, lorentzian};
use morphis::ops::{geometric, interior_left, interior_right, inverse, wedge};
use morphis::vector::basis;

// =============================================================================
// Wedge Product Tests
// =============================================================================

#[test]
fn wedge_basis_vectors() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let b = wedge(&e[0], &e[1]);

    assert_eq!(b.grade(), 2);
    assert_eq!(b.component(&[0, 1]), 1.0);
    assert_eq!(b.component(&[1, 0]), -1.0);
    assert_eq!(b.component(&[0, 0]), 0.0);
}

#[test]
fn wedge_anticommutativity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let u = &(&e[0] * 2.0) + &(&e[1] * 3.0);
    let v = &(&e[1] * 1.0) + &(&e[2] * 4.0);

    let uv = wedge(&u, &v);
    let vu = wedge(&v, &u);
    let neg_vu = -&vu;

    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (uv.component(&[m, n]) - neg_vu.component(&[m, n])).abs() < 1e-12,
                "anticommutativity failed at [{}, {}]",
                m,
                n,
            );
        }
    }
}

#[test]
fn wedge_nilpotency() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // e_m ^ e_m == 0 for all m
    for m in 0..3 {
        let b = wedge(&e[m], &e[m]);
        assert!(b.is_zero(1e-12), "e_{} ^ e_{} should be zero", m, m);
    }
}

#[test]
fn wedge_grade_exceeds_dim() {
    let g: Metric<2> = euclidean();
    let e = basis(g);

    let b = wedge(&e[0], &e[1]);
    let trivec = wedge(&b, &e[0]);

    assert_eq!(trivec.grade(), 3);
    assert!(trivec.is_zero(1e-12));
}

#[test]
fn wedge_three_basis_vectors() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let b01 = wedge(&e[0], &e[1]);
    let trivec = wedge(&b01, &e[2]);

    assert_eq!(trivec.grade(), 3);
    assert_eq!(trivec.component(&[0, 1, 2]), 1.0);
    assert_eq!(trivec.component(&[1, 0, 2]), -1.0);
    assert_eq!(trivec.component(&[0, 2, 1]), -1.0);
}

#[test]
fn wedge_with_scalar() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let s = morphis::vector::Vector::<3>::scalar(3.0, g);
    let v = wedge(&s, &e[1]);

    assert_eq!(v.grade(), 1);
    assert_eq!(v.component(&[1]), 3.0);
}

// =============================================================================
// Geometric Product Tests
// =============================================================================

#[test]
fn geometric_basis_vectors_euclidean() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // e_0 * e_0 = 1 (Euclidean)
    let product = geometric(&e[0], &e[0]);
    let s = product.grade_select(0).unwrap();
    assert!((s.data[IxDyn(&[])] - 1.0).abs() < 1e-12);
}

#[test]
fn geometric_product_decomposition() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // For grade-1 vectors: u * v = (u . v) + (u ^ v)
    let u = &(&e[0] * 2.0) + &(&e[1] * 1.0);
    let v = &(&e[0] * 1.0) + &(&e[2] * 3.0);

    let product = geometric(&u, &v);

    // Scalar part = dot product = 2*1 + 1*0 + 0*3 = 2
    let scalar = product.grade_select(0).unwrap();
    assert!((scalar.data[IxDyn(&[])] - 2.0).abs() < 1e-12);

    // Bivector part = wedge product
    let bv = product.grade_select(2).unwrap();
    let w = wedge(&u, &v);
    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (bv.component(&[m, n]) - w.component(&[m, n])).abs() < 1e-12,
                "geometric bivector != wedge at [{}, {}]",
                m,
                n,
            );
        }
    }
}

#[test]
fn geometric_signature_lorentzian() {
    let g: Metric<4> = lorentzian();
    let e = basis(g);

    // e_0 * e_0 = +1 (timelike)
    let p0 = geometric(&e[0], &e[0]);
    let s0 = p0.grade_select(0).unwrap();
    assert!((s0.data[IxDyn(&[])] - 1.0).abs() < 1e-12);

    // e_1 * e_1 = -1 (spacelike)
    let p1 = geometric(&e[1], &e[1]);
    let s1 = p1.grade_select(0).unwrap();
    assert!((s1.data[IxDyn(&[])] + 1.0).abs() < 1e-12);
}

// =============================================================================
// Interior Product Tests
// =============================================================================

#[test]
fn interior_left_vector_bivector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // e_0 ⌋ (e_0 ^ e_1) should give e_1
    let b = wedge(&e[0], &e[1]);
    let result = interior_left(&e[0], &b);

    assert_eq!(result.grade(), 1);
    assert!((result.component(&[0]) - 0.0).abs() < 1e-12);
    assert!((result.component(&[1]) - 1.0).abs() < 1e-12);
    assert!((result.component(&[2]) - 0.0).abs() < 1e-12);
}

#[test]
fn interior_right_bivector_vector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // (e_0 ^ e_1) ⌊ e_1 should give e_0
    let b = wedge(&e[0], &e[1]);
    let result = interior_right(&b, &e[1]);

    assert_eq!(result.grade(), 1);
    assert!((result.component(&[0]) - 1.0).abs() < 1e-12);
    assert!((result.component(&[1]) - 0.0).abs() < 1e-12);
}

#[test]
fn interior_left_grade_too_high() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let b = wedge(&e[0], &e[1]);
    // bivector ⌋ vector: grade 2 > grade 1, returns zero scalar
    let result = interior_left(&b, &e[0]);

    assert_eq!(result.grade(), 0);
    assert!((result.data[IxDyn(&[])]).abs() < 1e-12);
}

// =============================================================================
// Operator Syntax Tests
// =============================================================================

#[test]
fn wedge_operator() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let e12 = e[0].clone() ^ e[1].clone();

    assert_eq!(e12.grade(), 2);
    assert_eq!(e12.component(&[0, 1]), 1.0);
    assert_eq!(e12.component(&[1, 0]), -1.0);
}

#[test]
fn wedge_operator_chained() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let e123 = e[0].clone() ^ e[1].clone() ^ e[2].clone();

    assert_eq!(e123.grade(), 3);
    assert_eq!(e123.component(&[0, 1, 2]), 1.0);
    assert_eq!(e123.component(&[1, 0, 2]), -1.0);
}

#[test]
fn geometric_product_operator() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let product = e[0].clone() * e[0].clone();

    let s = product.grade_select(0).unwrap();
    assert!((s.data[IxDyn(&[])] - 1.0).abs() < 1e-12);
}

#[test]
fn interior_left_operator() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let b = e[0].clone() ^ e[1].clone();
    let result = e[0].clone() << b;

    assert_eq!(result.grade(), 1);
    assert!((result.component(&[1]) - 1.0).abs() < 1e-12);
}

#[test]
fn interior_right_operator() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let b = e[0].clone() ^ e[1].clone();
    let result = b >> e[1].clone();

    assert_eq!(result.grade(), 1);
    assert!((result.component(&[0]) - 1.0).abs() < 1e-12);
}

// =============================================================================
// Inverse Tests
// =============================================================================

#[test]
fn inverse_basis_vector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // e_0^{-1} = e_0 (unit vector in Euclidean)
    let e0_inv = inverse(&e[0]).unwrap();
    assert_eq!(e0_inv.grade(), 1);
    assert!((e0_inv.component(&[0]) - 1.0).abs() < 1e-12);
}

#[test]
fn inverse_scaled_vector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &e[0] * 3.0;
    let v_inv = inverse(&v).unwrap();

    // v * v^{-1} should be scalar 1
    let product = geometric(&v, &v_inv);
    let s = product.grade_select(0).unwrap();
    assert!((s.data[IxDyn(&[])] - 1.0).abs() < 1e-12);
}

// =============================================================================
// Algebraic Law Tests
// =============================================================================

#[test]
fn reverse_involution() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let u = &(&e[0] * 2.0) + &(&e[1] * 3.0);
    let b = wedge(&u, &e[2]);

    // rev(rev(v)) == v for any grade
    let b_rev_rev = b.rev().rev();
    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (b.component(&[m, n]) - b_rev_rev.component(&[m, n])).abs() < 1e-12,
                "reverse involution failed at [{}, {}]",
                m,
                n,
            );
        }
    }
}

#[test]
fn geometric_product_grade_1_decomposition() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // For any grade-1 vectors a, b:
    //   a * b == (a . b) + (a ^ b)
    // where a . b = a ⌋ b (left interior for equal grades)
    let a = &(&e[0] * 1.0) + &(&(&e[1] * 2.0) + &(&e[2] * 3.0));
    let b = &(&e[0] * 4.0) + &(&(&e[1] * 5.0) + &(&e[2] * 6.0));

    let product = geometric(&a, &b);
    let w = wedge(&a, &b);

    // Scalar part of geometric = dot product = 1*4 + 2*5 + 3*6 = 32
    let scalar = product.grade_select(0).unwrap();
    assert!((scalar.data[IxDyn(&[])] - 32.0).abs() < 1e-12);

    // Bivector part of geometric = wedge product
    let bv = product.grade_select(2).unwrap();
    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (bv.component(&[m, n]) - w.component(&[m, n])).abs() < 1e-12,
                "decomposition failed at [{}, {}]",
                m,
                n,
            );
        }
    }
}
