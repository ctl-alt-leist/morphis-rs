use morphis::metric::{Metric, euclidean};
use morphis::vector::{Vector, basis, basis_element, basis_vector, pseudoscalar};

#[test]
fn basis_vectors_3d() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    assert_eq!(e[0].component(&[0]), 1.0);
    assert_eq!(e[0].component(&[1]), 0.0);
    assert_eq!(e[0].component(&[2]), 0.0);

    assert_eq!(e[1].component(&[0]), 0.0);
    assert_eq!(e[1].component(&[1]), 1.0);

    assert_eq!(e[2].component(&[2]), 1.0);
}

#[test]
fn zero_vector() {
    let g: Metric<3> = euclidean();
    let v = Vector::<3>::zero(1, g);

    assert!(v.is_zero(1e-15));
    assert_eq!(v.grade(), 1);
    assert_eq!(v.dim(), 3);
}

#[test]
fn scalar_construction() {
    let g: Metric<3> = euclidean();
    let s = Vector::<3>::scalar(5.0, g);

    assert_eq!(s.grade(), 0);
    assert_eq!(s.scalar_value(), 5.0);
}

#[test]
fn vector_addition() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &e[0] + &e[1];

    assert_eq!(v.component(&[0]), 1.0);
    assert_eq!(v.component(&[1]), 1.0);
    assert_eq!(v.component(&[2]), 0.0);
}

#[test]
fn vector_subtraction() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &e[1] - &e[0];

    assert_eq!(v.component(&[0]), -1.0);
    assert_eq!(v.component(&[1]), 1.0);
}

#[test]
fn vector_negation() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = -&e[0];

    assert_eq!(v.component(&[0]), -1.0);
}

#[test]
fn scalar_multiplication() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &e[0] * 3.0;

    assert_eq!(v.component(&[0]), 3.0);
}

#[test]
fn euclidean_norm_grade_1() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &(&e[0] * 3.0) + &(&e[1] * 4.0);

    assert!((v.norm() - 5.0).abs() < 1e-12);
}

#[test]
fn normalize_unit_vector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &(&e[0] * 3.0) + &(&e[1] * 4.0);
    let u = v.normalize().unwrap();

    assert!((u.norm() - 1.0).abs() < 1e-12);
    assert!((u.component(&[0]) - 0.6).abs() < 1e-12);
    assert!((u.component(&[1]) - 0.8).abs() < 1e-12);
}

#[test]
fn reverse_involution() {
    let g: Metric<3> = euclidean();
    // Create a bivector: b = e0 ^ e1
    let e = basis(g);
    let b = morphis::ops::wedge(&e[0], &e[1]);

    // Grade-2: rev sign = (-1)^{2*1/2} = -1
    let b_rev = b.rev();
    assert_eq!(b_rev.component(&[0, 1]), -1.0);
    assert_eq!(b_rev.component(&[1, 0]), 1.0);

    // Double reverse restores original
    let b_rev_rev = b_rev.rev();
    assert_eq!(b_rev_rev, b);
}

#[test]
fn reverse_grade_1_is_identity() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // Grade-1: rev sign = (-1)^0 = +1
    let e0_rev = e[0].rev();
    assert_eq!(e0_rev, e[0]);
}

#[test]
fn basis_vector_construction() {
    let g: Metric<3> = euclidean();

    let e1 = basis_vector(0, g);
    assert_eq!(e1.grade(), 1);
    assert_eq!(e1.component(&[0]), 1.0);
    assert_eq!(e1.component(&[1]), 0.0);
}

#[test]
fn basis_element_bivector() {
    let g: Metric<3> = euclidean();

    let e12 = basis_element(&[0, 1], g);
    assert_eq!(e12.grade(), 2);
    assert_eq!(e12.component(&[0, 1]), 1.0);
    assert_eq!(e12.component(&[1, 0]), -1.0);
}

#[test]
fn basis_element_empty_is_scalar() {
    let g: Metric<3> = euclidean();

    let s = basis_element(&[], g);
    assert_eq!(s.grade(), 0);
    assert!((s.scalar_value() - 1.0).abs() < 1e-12);
}

#[test]
fn pseudoscalar_3d() {
    let g: Metric<3> = euclidean();

    let ps = pseudoscalar(g);
    assert_eq!(ps.grade(), 3);
    assert_eq!(ps.component(&[0, 1, 2]), 1.0);
}

#[test]
fn inv_method() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = &e[0] * 3.0;
    let v_inv = v.inv().unwrap();

    assert!((v_inv.component(&[0]) - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn owned_arithmetic() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let v = e[0].clone() + e[1].clone();
    assert_eq!(v.component(&[0]), 1.0);
    assert_eq!(v.component(&[1]), 1.0);

    let w = e[0].clone() - e[1].clone();
    assert_eq!(w.component(&[0]), 1.0);
    assert_eq!(w.component(&[1]), -1.0);

    let scaled = 3.0 * e[2].clone();
    assert_eq!(scaled.component(&[2]), 3.0);
}

#[test]
fn reverse_grade_3_sign() {
    let g: Metric<3> = euclidean();
    let ps = pseudoscalar(g);

    // Grade-3: rev sign = (-1)^{3*2/2} = (-1)^3 = -1
    let ps_rev = ps.rev();
    assert_eq!(ps_rev.component(&[0, 1, 2]), -1.0);
}

#[test]
fn antisymmetric_access() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // Bivector b = e0 ^ e1
    let b = morphis::ops::wedge(&e[0], &e[1]);

    // Canonical component
    assert_eq!(b.component(&[0, 1]), 1.0);
    // Transposed: should be negative
    assert_eq!(b.component(&[1, 0]), -1.0);
    // Repeated: should be zero
    assert_eq!(b.component(&[0, 0]), 0.0);
    assert_eq!(b.component(&[1, 1]), 0.0);
}

#[test]
fn sparse_storage_size() {
    let g: Metric<3> = euclidean();

    // Grade-1 in D=3: C(3,1) = 3 components
    let v = Vector::<3>::zero(1, g);
    assert_eq!(v.n_components(), 3);

    // Grade-2 in D=3: C(3,2) = 3 components (was 9 dense)
    let b = Vector::<3>::zero(2, g);
    assert_eq!(b.n_components(), 3);

    // Grade-3 in D=3: C(3,3) = 1 component (was 27 dense)
    let t = Vector::<3>::zero(3, g);
    assert_eq!(t.n_components(), 1);

    // Grade-2 in D=4: C(4,2) = 6 components (was 16 dense)
    let g4: Metric<4> = euclidean();
    let b4 = Vector::<4>::zero(2, g4);
    assert_eq!(b4.n_components(), 6);
}
