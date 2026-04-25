use morphis::metric::{Metric, Signature, euclidean, lorentzian, projective};

#[test]
fn euclidean_metric_3d() {
    let g: Metric<3> = euclidean();
    assert_eq!(g.dim(), 3);
    assert_eq!(g.sig, Signature::Euclidean);
    assert_eq!(g.diag, [1.0, 1.0, 1.0]);
}

#[test]
fn lorentzian_metric_4d() {
    let g: Metric<4> = lorentzian();
    assert_eq!(g.dim(), 4);
    assert_eq!(g.sig, Signature::Lorentzian);
    assert_eq!(g.diag, [1.0, -1.0, -1.0, -1.0]);
}

#[test]
fn projective_metric_4d() {
    let g: Metric<4> = projective();
    assert_eq!(g.dim(), 4);
    assert_eq!(g.sig, Signature::Projective);
    assert_eq!(g.diag, [0.0, 1.0, 1.0, 1.0]);
}

#[test]
fn metric_component_access() {
    let g: Metric<3> = euclidean();
    assert_eq!(g.component(0, 0), 1.0);
    assert_eq!(g.component(1, 2), 0.0);
    assert_eq!(g.component(2, 2), 1.0);
}

#[test]
fn lorentzian_component_signs() {
    let g: Metric<4> = lorentzian();
    assert_eq!(g.component(0, 0), 1.0);
    assert_eq!(g.component(1, 1), -1.0);
    assert_eq!(g.component(2, 2), -1.0);
    assert_eq!(g.component(3, 3), -1.0);
}
