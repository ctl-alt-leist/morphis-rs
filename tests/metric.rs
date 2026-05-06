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

// =============================================================================
// Physics-Convention Index Translation
// =============================================================================

#[test]
fn base_index_euclidean() {
    let g: Metric<3> = euclidean();
    assert_eq!(g.base_index(), 1);
    assert_eq!(g.max_index(), 3);
}

#[test]
fn base_index_lorentzian() {
    let g: Metric<4> = lorentzian();
    assert_eq!(g.base_index(), 0);
    assert_eq!(g.max_index(), 3);
}

#[test]
fn base_index_projective() {
    let g: Metric<4> = projective();
    assert_eq!(g.base_index(), 1);
    assert_eq!(g.max_index(), 4);
}

#[test]
fn to_internal_euclidean() {
    let g: Metric<3> = euclidean();
    assert_eq!(g.to_internal(1), 0);
    assert_eq!(g.to_internal(2), 1);
    assert_eq!(g.to_internal(3), 2);
}

#[test]
fn to_internal_lorentzian() {
    let g: Metric<4> = lorentzian();
    // Timelike: index 0 → internal 0
    assert_eq!(g.to_internal(0), 0);
    // Spatial: index 1 → internal 1
    assert_eq!(g.to_internal(1), 1);
    assert_eq!(g.to_internal(3), 3);
}

#[test]
fn to_user_roundtrip() {
    let g_euc: Metric<3> = euclidean();
    for k in 0..3 {
        assert_eq!(g_euc.to_internal(g_euc.to_user(k)), k);
    }

    let g_lor: Metric<4> = lorentzian();
    for k in 0..4 {
        assert_eq!(g_lor.to_internal(g_lor.to_user(k)), k);
    }
}

#[test]
fn to_internal_multi_euclidean() {
    let g: Metric<3> = euclidean();
    assert_eq!(g.to_internal_multi(&[1, 2]), vec![0, 1]);
    assert_eq!(g.to_internal_multi(&[2, 3]), vec![1, 2]);
}

#[test]
fn x_component_is_always_index_1() {
    // The first spatial direction is always user-facing index 1,
    // regardless of signature.
    let g_euc: Metric<3> = euclidean();
    assert_eq!(g_euc.to_internal(1), 0); // index 1 → first stored component

    let g_lor: Metric<4> = lorentzian();
    assert_eq!(g_lor.to_internal(1), 1); // index 1 → second stored (first spatial)
}

#[test]
#[should_panic(expected = "out of range")]
fn euclidean_index_zero_panics() {
    let g: Metric<3> = euclidean();
    g.to_internal(0);
}

#[test]
#[should_panic(expected = "out of range")]
fn euclidean_index_too_high_panics() {
    let g: Metric<3> = euclidean();
    g.to_internal(4); // valid range is 1..=3
}

#[test]
#[should_panic(expected = "out of range")]
fn lorentzian_index_too_high_panics() {
    let g: Metric<4> = lorentzian();
    g.to_internal(4); // valid range is 0..=3
}
