use std::collections::HashMap;

use morphis::metric::{Metric, euclidean};
use morphis::multivector::MultiVector;
use morphis::ops::wedge;
use morphis::vector::{Vector, basis};

#[test]
fn from_vector() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let mv = MultiVector::from_vector(e[1].clone());

    assert_eq!(mv.grades(), vec![1]);
    assert!(mv.grade_select(1).is_some());
    assert!(mv.grade_select(0).is_none());
}

#[test]
fn multivector_addition() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let s = Vector::<3>::scalar(5.0, g);
    let mv1 = MultiVector::from_vector(s);
    let mv2 = MultiVector::from_vector(e[1].clone());

    let sum = &mv1 + &mv2;

    assert_eq!(sum.grades(), vec![0, 1]);
}

#[test]
fn multivector_reverse() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let b = wedge(&e[1], &e[2]);
    let s = Vector::<3>::scalar(1.0, g);

    let mut components = HashMap::new();
    components.insert(0, s);
    components.insert(2, b);
    let mv = MultiVector::from_components(components, g);

    let mv_rev = mv.rev();

    // Scalar part unchanged
    let s_rev = mv_rev.grade_select(0).unwrap();
    assert!((s_rev.scalar_value() - 1.0).abs() < 1e-12);

    // Bivector part negated (grade-2 reversal sign = -1)
    let b_rev = mv_rev.grade_select(2).unwrap();
    assert!((b_rev.component(&[1, 2]) + 1.0).abs() < 1e-12);
}

#[test]
fn is_even_rotor_like() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let s = Vector::<3>::scalar(1.0, g);
    let b = wedge(&e[1], &e[2]);

    let mut components = HashMap::new();
    components.insert(0, s);
    components.insert(2, b);
    let mv = MultiVector::from_components(components, g);

    assert!(mv.is_even());
}

#[test]
fn scalar_multiplication() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let mv = MultiVector::from_vector(e[1].clone());
    let scaled = &mv * 3.0;

    let v = scaled.grade_select(1).unwrap();
    assert!((v.component(&[1]) - 3.0).abs() < 1e-12);
}
