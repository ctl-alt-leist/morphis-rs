use morphis::metric::{Metric, euclidean, lorentzian};
use morphis::multivector::MultiVector;
use morphis::ops::{
    geometric, geometric_mv_mv, geometric_mv_v, geometric_v_mv, interior_left, interior_right,
    inverse, project, reflect, reject, wedge,
};
use morphis::vector::{Vector, basis};

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

    let s = Vector::<3>::scalar(3.0, g);
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
    assert!((s.scalar_value() - 1.0).abs() < 1e-12);
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
    assert!((scalar.scalar_value() - 2.0).abs() < 1e-12);

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
    assert!((s0.scalar_value() - 1.0).abs() < 1e-12);

    // e_1 * e_1 = -1 (spacelike)
    let p1 = geometric(&e[1], &e[1]);
    let s1 = p1.grade_select(0).unwrap();
    assert!((s1.scalar_value() + 1.0).abs() < 1e-12);
}

// =============================================================================
// Interior Product Tests
// =============================================================================

#[test]
fn interior_left_vector_bivector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // e_0 . (e_0 ^ e_1) should give e_1
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

    // (e_0 ^ e_1) . e_1 should give e_0
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
    // bivector . vector: grade 2 > grade 1, returns zero scalar
    let result = interior_left(&b, &e[0]);

    assert_eq!(result.grade(), 0);
    assert!(result.scalar_value().abs() < 1e-12);
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
    assert!((s.scalar_value() - 1.0).abs() < 1e-12);
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
    assert!((s.scalar_value() - 1.0).abs() < 1e-12);
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
    let a = &(&e[0] * 1.0) + &(&(&e[1] * 2.0) + &(&e[2] * 3.0));
    let b = &(&e[0] * 4.0) + &(&(&e[1] * 5.0) + &(&e[2] * 6.0));

    let product = geometric(&a, &b);
    let w = wedge(&a, &b);

    // Scalar part of geometric = dot product = 1*4 + 2*5 + 3*6 = 32
    let scalar = product.grade_select(0).unwrap();
    assert!((scalar.scalar_value() - 32.0).abs() < 1e-12);

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

#[test]
fn geometric_product_associativity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let a = &(&e[0] * 2.0) + &(&e[1] * 1.0);
    let b = &(&e[1] * 3.0) + &(&e[2] * 1.0);
    let c = &(&e[0] * 1.0) + &(&e[2] * 2.0);

    // (a * b) * c = a * (b * c)
    let ab = geometric(&a, &b);
    let left = &ab * &c;

    let bc = geometric(&b, &c);
    let right = &a * &bc;

    for k in 0..=3 {
        let lk = left.grade_project(k);
        let rk = right.grade_project(k);
        assert!(
            (&lk - &rk).is_zero(1e-11),
            "associativity failed at grade {}",
            k,
        );
    }
}

#[test]
fn wedge_associativity() {
    let g: Metric<4> = euclidean();
    let e = basis(g);

    let u = &(&e[0] * 2.0) + &(&e[1] * 1.0);
    let v = &(&e[1] * 3.0) + &(&e[2] * 1.0);
    let w = &(&e[2] * 1.0) + &(&e[3] * 2.0);

    // (u ^ v) ^ w = u ^ (v ^ w)
    let left = wedge(&wedge(&u, &v), &w);
    let right = wedge(&u, &wedge(&v, &w));

    assert_eq!(left.grade(), 3);
    assert_eq!(right.grade(), 3);
    assert!(
        (&left - &right).is_zero(1e-12),
        "wedge associativity failed",
    );
}

#[test]
fn product_reversion_law() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // rev(u * v) = rev(v) * rev(u)
    let u = &(&e[0] * 2.0) + &(&e[1] * 3.0);
    let v = &(&e[1] * 1.0) + &(&e[2] * 4.0);

    let product = geometric(&u, &v);
    let left = product.rev();

    let right = geometric(&v.rev(), &u.rev());

    for k in 0..=3 {
        let lk = left.grade_project(k);
        let rk = right.grade_project(k);
        assert!(
            (&lk - &rk).is_zero(1e-12),
            "product reversion failed at grade {}",
            k,
        );
    }
}

#[test]
fn bivector_inverse() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // B * B^{-1} = 1 for a bivector
    let b = wedge(&e[0], &e[1]);
    let b_inv = inverse(&b).unwrap();

    let product = geometric(&b, &b_inv);
    assert!(
        (product.scalar_part() - 1.0).abs() < 1e-12,
        "B * B^-1 scalar should be 1, got {}",
        product.scalar_part(),
    );

    // Non-scalar grades should vanish
    for k in 1..=3 {
        assert!(
            product.grade_project(k).is_zero(1e-12),
            "B * B^-1 should have no grade-{} component",
            k,
        );
    }
}

// =============================================================================
// Projection, Rejection, Reflection Tests
// =============================================================================

#[test]
fn project_onto_basis_vector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &(&e[0] * 3.0) + &(&e[1] * 4.0);
    let p = project(&v, &e[0]);

    assert_eq!(p.grade(), 1);
    assert!((p.component(&[0]) - 3.0).abs() < 1e-12);
    assert!(p.component(&[1]).abs() < 1e-12);
    assert!(p.component(&[2]).abs() < 1e-12);
}

#[test]
fn project_onto_general_vector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let target = &e[0] + &e[1];
    let p = project(&e[0], &target);

    assert!((p.component(&[0]) - 0.5).abs() < 1e-12);
    assert!((p.component(&[1]) - 0.5).abs() < 1e-12);
    assert!(p.component(&[2]).abs() < 1e-12);
}

#[test]
fn reject_from_basis_vector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &(&e[0] * 3.0) + &(&e[1] * 4.0);
    let r = reject(&v, &e[0]);

    assert!(r.component(&[0]).abs() < 1e-12);
    assert!((r.component(&[1]) - 4.0).abs() < 1e-12);
}

#[test]
fn project_plus_reject_equals_original() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));
    let target = &(&e[0] * 1.0) + &(&e[1] * 1.0);

    let p = project(&v, &target);
    let r = reject(&v, &target);
    let sum = &p + &r;

    for m in 0..3 {
        assert!(
            (sum.component(&[m]) - v.component(&[m])).abs() < 1e-12,
            "project + reject should equal original at component {}",
            m,
        );
    }
}

#[test]
fn projection_is_idempotent() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));
    let target = &(&e[0] * 1.0) + &(&e[2] * 1.0);

    let p1 = project(&v, &target);
    let p2 = project(&p1, &target);

    for m in 0..3 {
        assert!(
            (p1.component(&[m]) - p2.component(&[m])).abs() < 1e-12,
            "projection should be idempotent at component {}",
            m,
        );
    }
}

#[test]
fn rejection_is_orthogonal() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));
    let target = &(&e[0] * 1.0) + &(&e[1] * 1.0);

    let r = reject(&v, &target);

    // r dot target should be zero
    let dot = interior_left(&r, &target);
    assert!(
        dot.scalar_value().abs() < 1e-12,
        "rejection should be orthogonal to target, got dot = {}",
        dot.scalar_value(),
    );
}

#[test]
fn reflect_through_e1() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &(&e[0] * 2.0) + &(&e[1] * 3.0);
    let r = reflect(&v, &e[0]);

    assert!((r.component(&[0]) + 2.0).abs() < 1e-12);
    assert!((r.component(&[1]) - 3.0).abs() < 1e-12);
    assert!(r.component(&[2]).abs() < 1e-12);
}

#[test]
fn reflect_twice_is_identity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));
    let n = &(&e[0] * 1.0) + &(&e[1] * 1.0);

    let once = reflect(&v, &n);
    let twice = reflect(&once, &n);

    for m in 0..3 {
        assert!(
            (twice.component(&[m]) - v.component(&[m])).abs() < 1e-11,
            "double reflection should be identity at component {}",
            m,
        );
    }
}

#[test]
fn reflect_preserves_norm() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));
    let n = &e[1];

    let r = reflect(&v, n);

    assert!(
        (r.norm() - v.norm()).abs() < 1e-12,
        "reflection should preserve norm: {} vs {}",
        r.norm(),
        v.norm(),
    );
}

// =============================================================================
// MultiVector Product Tests
// =============================================================================

#[test]
fn mv_times_vector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let s = Vector::<3>::scalar(1.0, g);
    let mv = MultiVector::from_vector(s) + MultiVector::from_vector(e[0].clone());

    let result = geometric_mv_v(&mv, &e[0]);

    assert!(
        (result.scalar_part() - 1.0).abs() < 1e-12,
        "scalar part should be 1, got {}",
        result.scalar_part(),
    );
    let v = result.grade_project(1);
    assert!(
        (v.component(&[0]) - 1.0).abs() < 1e-12,
        "e1 component should be 1, got {}",
        v.component(&[0]),
    );
}

#[test]
fn vector_times_mv() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let s = Vector::<3>::scalar(1.0, g);
    let mv = MultiVector::from_vector(s) + MultiVector::from_vector(e[1].clone());

    let result = geometric_v_mv(&e[0], &mv);

    let v = result.grade_project(1);
    assert!((v.component(&[0]) - 1.0).abs() < 1e-12);

    let bv = result.grade_project(2);
    assert!((bv.component(&[0, 1]) - 1.0).abs() < 1e-12);
}

#[test]
fn mv_times_mv_rotor_product() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let cos_val = (std::f64::consts::PI / 4.0).cos();
    let sin_val = (std::f64::consts::PI / 4.0).sin();

    let b = wedge(&e[0], &e[1]);
    let r = MultiVector::from_vector(Vector::<3>::scalar(cos_val, g))
        + MultiVector::from_vector(&b * (-sin_val));

    let r_rev = r.rev();
    let product = geometric_mv_mv(&r, &r_rev);

    assert!(
        (product.scalar_part() - 1.0).abs() < 1e-12,
        "R ~R should be 1, got {}",
        product.scalar_part(),
    );
}

#[test]
fn mv_product_operator_syntax() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let mv1 = MultiVector::from_vector(e[0].clone());
    let mv2 = MultiVector::from_vector(e[0].clone());

    let product = &mv1 * &mv2;
    assert!((product.scalar_part() - 1.0).abs() < 1e-12);

    let result_1 = mv1.clone() * e[1].clone();
    let bv_1 = result_1.grade_project(2);
    assert!((bv_1.component(&[0, 1]) - 1.0).abs() < 1e-12);

    let result_2 = e[1].clone() * mv2;
    let bv_2 = result_2.grade_project(2);
    assert!((bv_2.component(&[0, 1]) + 1.0).abs() < 1e-12);
}
