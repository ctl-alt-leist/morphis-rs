use morphis::duality::{join, left_complement, meet, right_complement};
use morphis::metric::{Metric, euclidean};
use morphis::ops::wedge;
use morphis::vector::{Vector, basis, pseudoscalar};

// =============================================================================
// Right Complement: Grade Mapping
// =============================================================================

#[test]
fn right_complement_grade() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // Grade 1 → grade 2
    let dual = right_complement(&e[0]);
    assert_eq!(dual.grade(), 2);
}

#[test]
fn right_complement_basis_vectors_3d() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // In 3D Euclidean, I^{-1} = -I, so ⋆u = u ⌋ I^{-1} = -(u ⌋ I)
    // ⋆e1 = -(e2 ∧ e3)
    let dual_e1 = right_complement(&e[0]);
    assert!((dual_e1.component(&[1, 2]) + 1.0).abs() < 1e-12);
    assert!((dual_e1.component(&[2, 1]) - 1.0).abs() < 1e-12);
    assert!(dual_e1.component(&[0, 1]).abs() < 1e-12);
    assert!(dual_e1.component(&[0, 2]).abs() < 1e-12);

    // ⋆e3 = -(e1 ∧ e2)
    let dual_e3 = right_complement(&e[2]);
    assert!((dual_e3.component(&[0, 1]) + 1.0).abs() < 1e-12);
}

#[test]
fn right_complement_bivector_to_vector_3d() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // ⋆(e1 ∧ e2) = -e3 (because I^{-1} = -I in 3D Euclidean)
    let b12 = wedge(&e[0], &e[1]);
    let dual = right_complement(&b12);

    assert_eq!(dual.grade(), 1);
    assert!(dual.component(&[0]).abs() < 1e-12);
    assert!(dual.component(&[1]).abs() < 1e-12);
    assert!((dual.component(&[2]) + 1.0).abs() < 1e-12);
}

#[test]
fn right_complement_scalar_to_pseudoscalar() {
    let g: Metric<3> = euclidean();

    let one = Vector::<3>::scalar(1.0, g);
    let dual = right_complement(&one);

    assert_eq!(dual.grade(), 3);

    // ⋆1 = 1 ⌋ I^{-1} = I^{-1} = -I in 3D Euclidean
    let ps = pseudoscalar(g);
    assert!(
        (dual.component(&[0, 1, 2]) + ps.component(&[0, 1, 2])).abs() < 1e-12,
        "⋆1 should be -I (since I^{{-1}} = -I in 3D Euclidean)",
    );
}

#[test]
fn right_complement_pseudoscalar_to_scalar() {
    let g: Metric<3> = euclidean();

    let ps = pseudoscalar(g);
    let dual = right_complement(&ps);

    assert_eq!(dual.grade(), 0);
    assert!(
        (dual.data[ndarray::IxDyn(&[])] - 1.0).abs() < 1e-12,
        "⋆I should be 1, got {}",
        dual.data[ndarray::IxDyn(&[])],
    );
}

// =============================================================================
// Double Dual
// =============================================================================

#[test]
fn double_right_complement_vectors_3d() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // In 3D Euclidean: ⋆⋆v = (-1)^{k(d-k)} v = (-1)^{1·2} v = v
    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));
    let double_dual = right_complement(&right_complement(&v));

    for m in 0..3 {
        assert!(
            (double_dual.component(&[m]) - v.component(&[m])).abs() < 1e-12,
            "⋆⋆v should equal v in 3D Euclidean at component {}",
            m,
        );
    }
}

#[test]
fn double_right_complement_bivector_3d() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // ⋆⋆B = (-1)^{2·1} B = B
    let b = wedge(&e[0], &e[1]);
    let double_dual = right_complement(&right_complement(&b));

    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (double_dual.component(&[m, n]) - b.component(&[m, n])).abs() < 1e-12,
                "⋆⋆B should equal B at [{}, {}]",
                m,
                n,
            );
        }
    }
}

// =============================================================================
// Left Complement
// =============================================================================

#[test]
fn left_complement_grade() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    let dual = left_complement(&e[0]);
    assert_eq!(dual.grade(), 2);
}

// =============================================================================
// Higher Dimension
// =============================================================================

#[test]
fn right_complement_4d() {
    let g: Metric<4> = euclidean();
    let e = basis(g);

    // In 4D: ⋆e1 should be grade 3
    let dual = right_complement(&e[0]);
    assert_eq!(dual.grade(), 3);

    // ⋆(e1 ∧ e2) should be grade 2
    let b = wedge(&e[0], &e[1]);
    let dual_b = right_complement(&b);
    assert_eq!(dual_b.grade(), 2);
}

#[test]
fn double_right_complement_4d_vector() {
    let g: Metric<4> = euclidean();
    let e = basis(g);

    // In 4D Euclidean: ⋆⋆v for grade 1 = (-1)^{1·3} v = -v
    let v = &(&e[0] * 2.0) + &(&(&e[1] * 3.0) + &(&e[2] * 5.0));
    let double_dual = right_complement(&right_complement(&v));

    for m in 0..4 {
        assert!(
            (double_dual.component(&[m]) + v.component(&[m])).abs() < 1e-12,
            "⋆⋆v should equal -v in 4D Euclidean at component {}",
            m,
        );
    }
}

#[test]
fn double_right_complement_4d_bivector() {
    let g: Metric<4> = euclidean();
    let e = basis(g);

    // In 4D Euclidean: ⋆⋆B for grade 2 = (-1)^{2·2} B = B
    let b = wedge(&e[0], &e[1]);
    let double_dual = right_complement(&right_complement(&b));

    for m in 0..4 {
        for n in 0..4 {
            assert!(
                (double_dual.component(&[m, n]) - b.component(&[m, n])).abs() < 1e-12,
                "⋆⋆B should equal B in 4D at [{}, {}]",
                m,
                n,
            );
        }
    }
}

// =============================================================================
// Meet and Join
// =============================================================================

#[test]
fn join_independent_vectors() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // join(e1, e2) = e1 ∧ e2 (independent vectors)
    let result = join(&e[0], &e[1]);
    let expected = wedge(&e[0], &e[1]);

    assert_eq!(result.grade(), 2);
    for m in 0..3 {
        for n in 0..3 {
            assert!(
                (result.component(&[m, n]) - expected.component(&[m, n])).abs() < 1e-12,
                "join of independent vectors should equal wedge at [{}, {}]",
                m,
                n,
            );
        }
    }
}

#[test]
fn meet_planes_sharing_line() {
    let g: Metric<3> = euclidean();
    let e = basis(g);

    // Two planes sharing e1:
    // A = e1 ∧ e2 (the e1-e2 plane)
    // B = e1 ∧ e3 (the e1-e3 plane)
    // meet should be proportional to e1
    let a = wedge(&e[0], &e[1]);
    let b = wedge(&e[0], &e[2]);

    let result = meet(&a, &b);

    assert_eq!(result.grade(), 1);

    // Should be proportional to e1
    let e2_component = result.component(&[1]);
    let e3_component = result.component(&[2]);
    assert!(
        e2_component.abs() < 1e-12,
        "meet should have no e2 component, got {}",
        e2_component,
    );
    assert!(
        e3_component.abs() < 1e-12,
        "meet should have no e3 component, got {}",
        e3_component,
    );
    assert!(
        result.component(&[0]).abs() > 1e-12,
        "meet should have nonzero e1 component",
    );
}

#[test]
fn meet_disjoint_lines_is_zero() {
    let g: Metric<4> = euclidean();
    let e = basis(g);

    // Two lines sharing no common direction in 4D
    // A = e1 ∧ e2
    // B = e3 ∧ e4
    // Their duals are grade-2, and the wedge of two grade-2 elements in 4D
    // is grade-4 (the pseudoscalar), whose dual is grade 0 (scalar).
    // The meet should reflect that the lines don't intersect.
    let a = wedge(&e[0], &e[1]);
    let b = wedge(&e[2], &e[3]);

    let result = meet(&a, &b);
    assert_eq!(result.grade(), 0);
}
