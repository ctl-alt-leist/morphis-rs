use ndarray::{ArrayD, IxDyn};

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
    assert_eq!(s.data[IxDyn(&[])], 5.0);
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
    let mut data = ArrayD::zeros(IxDyn(&[3, 3]));
    data[[0, 1]] = 1.0;
    data[[1, 0]] = -1.0;
    let b = Vector::<3>::new(data, 2, g);

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
    assert!((s.data[IxDyn(&[])] - 1.0).abs() < 1e-12);
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
    let mut data = ArrayD::zeros(IxDyn(&[3, 3, 3]));
    data[[0, 1, 2]] = 1.0;
    data[[1, 0, 2]] = -1.0;
    data[[0, 2, 1]] = -1.0;
    data[[2, 1, 0]] = -1.0;
    data[[1, 2, 0]] = 1.0;
    data[[2, 0, 1]] = 1.0;
    let v = Vector::<3>::new(data, 3, g);

    // Grade-3: rev sign = (-1)^{3*2/2} = (-1)^3 = -1
    let v_rev = v.rev();
    assert_eq!(v_rev.component(&[0, 1, 2]), -1.0);
}
