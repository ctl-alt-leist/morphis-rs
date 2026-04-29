use std::f64::consts::PI;

use morphis::even_field::EvenField;
use morphis::field::Field;
use morphis::grid::Grid;
use morphis::metric::euclidean;

fn approx_eq(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() < tol,
        "values differ: {} vs {} (diff = {})",
        a,
        b,
        (a - b).abs()
    );
}

// =============================================================================
// EvenField Basic Properties
// =============================================================================

#[test]
fn reversal_is_involution() {
    // rev(rev(α)) = α
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(8, 1.0);

    let psi = EvenField::from_fn(&grid, g, |x| (x[0].cos(), x[1].sin()));

    let double_rev = psi.rev().rev();

    // Compare at several points
    for m in [0, 2, 4, 6] {
        let orig_s = psi.scalar[ndarray::IxDyn(&[m, 0, 0])];
        let orig_p = psi.pseudoscalar[ndarray::IxDyn(&[m, 0, 0])];
        let rev_s = double_rev.scalar[ndarray::IxDyn(&[m, 0, 0])];
        let rev_p = double_rev.pseudoscalar[ndarray::IxDyn(&[m, 0, 0])];
        approx_eq(orig_s, rev_s, 1e-12);
        approx_eq(orig_p, rev_p, 1e-12);
    }
}

#[test]
fn norm_squared_is_real() {
    // α_rev * α has zero pseudoscalar part (it's just a² + b²)
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(8, 1.0);

    let psi = EvenField::from_fn(&grid, g, |x| (x[0].cos(), x[1].sin() + 0.5));

    let norm_sq = psi.norm_squared();
    assert_eq!(norm_sq.grade(), 0);

    // Verify it's a² + b² at each point
    for m in [0, 2, 4, 6] {
        let a = psi.scalar[ndarray::IxDyn(&[m, 0, 0])];
        let b = psi.pseudoscalar[ndarray::IxDyn(&[m, 0, 0])];
        let expected = a * a + b * b;
        let actual = norm_sq.at(&[m, 0, 0]).component(&[]);
        approx_eq(actual, expected, 1e-12);
    }
}

#[test]
fn phase_rotation_preserves_norm() {
    // |exp(Iθ) α|² = |α|²
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(8, 1.0);

    let psi = EvenField::from_fn(&grid, g, |x| {
        ((2.0 * PI * x[0]).cos(), (2.0 * PI * x[1]).sin())
    });

    let angle = Field::scalar_field(&grid, g, |x| 0.7 * x[2]);
    let rotated = psi.rotate_phase(&angle);

    let norm_before = psi.norm_squared();
    let norm_after = rotated.norm_squared();

    for m in [0, 2, 4, 6] {
        for p in [0, 2, 4, 6] {
            let before = norm_before.at(&[m, p, 0]).component(&[]);
            let after = norm_after.at(&[m, p, 0]).component(&[]);
            approx_eq(before, after, 1e-12);
        }
    }
}

#[test]
fn product_closure() {
    // EvenField * EvenField stays in even subalgebra
    // (a + bI)(c + dI) = (ac - bd) + (ad + bc)I
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(4, 1.0);

    let f1 = EvenField::from_fn(&grid, g, |x| (x[0] + 1.0, x[1]));
    let f2 = EvenField::from_fn(&grid, g, |x| (x[2], x[0] + 0.5));

    let product = f1.mul(&f2);

    // Verify at a point
    let idx = [1, 2, 3];
    let a = f1.scalar[ndarray::IxDyn(&idx)];
    let b = f1.pseudoscalar[ndarray::IxDyn(&idx)];
    let c = f2.scalar[ndarray::IxDyn(&idx)];
    let d = f2.pseudoscalar[ndarray::IxDyn(&idx)];

    let expected_real = a * c - b * d;
    let expected_imag = a * d + b * c;
    approx_eq(product.scalar[ndarray::IxDyn(&idx)], expected_real, 1e-12);
    approx_eq(
        product.pseudoscalar[ndarray::IxDyn(&idx)],
        expected_imag,
        1e-12,
    );
}

#[test]
fn density_extraction() {
    // For α = sqrt(ρ/m) exp(I S), density(m) = m * |α|² = ρ
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(8, 1.0);
    let mass = 2.0;

    // Create a wavefunction with known density: ρ(x) = 1 + 0.5*cos(2πx)
    let psi = EvenField::from_fn(&grid, g, |x| {
        let rho = 1.0 + 0.5 * (2.0 * PI * x[0]).cos();
        let amplitude = (rho / mass).sqrt();
        let phase = 0.3 * x[1]; // arbitrary phase
        (amplitude * phase.cos(), amplitude * phase.sin())
    });

    let rho = psi.density(mass);

    for m in [0, 2, 4, 6] {
        let x = m as f64 * grid.cell_length;
        let expected = 1.0 + 0.5 * (2.0 * PI * x).cos();
        let actual = rho.at(&[m, 0, 0]).component(&[]);
        approx_eq(actual, expected, 1e-12);
    }
}

#[test]
fn at_extracts_multivector() {
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(4, 1.0);

    let psi = EvenField::from_fn(&grid, g, |_| (2.0, 3.0));
    let mv = psi.at(&[0, 0, 0]);

    // Should have grade-0 and grade-3 components
    let scalar_part = mv.scalar_part();
    approx_eq(scalar_part, 2.0, 1e-12);

    // Grade-3 part exists
    let grades = mv.grades();
    assert!(grades.contains(&0));
    assert!(grades.contains(&3));
}
