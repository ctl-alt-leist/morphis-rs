use std::f64::consts::PI;

use morphis::even_field::EvenField;
use morphis::field::Field;
use morphis::grid::Grid;
use morphis::metric::euclidean;
use ndarray::IxDyn;

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

// =============================================================================
// Norm Conservation
// =============================================================================

#[test]
fn norm_conservation_under_phase_rotation() {
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(16, 1.0);

    let psi = EvenField::from_fn(&grid, g, |x| {
        ((2.0 * PI * x[0]).cos() + 1.5, (2.0 * PI * x[1]).sin())
    });

    let angle = Field::scalar_field(&grid, g, |x| 1.3 * x[0] + 0.7 * x[2]);
    let rotated = psi.rotate_phase(&angle);

    approx_eq(
        psi.integrate_norm_squared(),
        rotated.integrate_norm_squared(),
        1e-12,
    );
}

// =============================================================================
// EvenField Laplacian
// =============================================================================

#[test]
fn laplacian_of_sinusoidal_scalar_component() {
    // α = (sin(2πx_0/L), 0) → ∇²α = (-(2π/L)² sin(...), 0)
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let psi = EvenField::from_fn(&grid, g, |x| ((k * x[0]).sin(), 0.0));
    let lap = psi.laplacian();

    for m in [0, 4, 8, 16, 24] {
        let x = m as f64 * grid.cell_length;
        approx_eq(lap.scalar[IxDyn(&[m, 0, 0])], -k * k * (k * x).sin(), 1e-10);
        approx_eq(lap.pseudoscalar[IxDyn(&[m, 0, 0])], 0.0, 1e-10);
    }
}

#[test]
fn laplacian_with_both_components() {
    // α = (cos(k x_0), sin(k x_1)) → acts independently
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let psi = EvenField::from_fn(&grid, g, |x| ((k * x[0]).cos(), (k * x[1]).sin()));
    let lap = psi.laplacian();

    for m in [0, 4, 8, 16, 24] {
        for p in [0, 4, 8, 16, 24] {
            let x0 = m as f64 * grid.cell_length;
            let x1 = p as f64 * grid.cell_length;
            approx_eq(
                lap.scalar[IxDyn(&[m, p, 0])],
                -k * k * (k * x0).cos(),
                1e-10,
            );
            approx_eq(
                lap.pseudoscalar[IxDyn(&[m, p, 0])],
                -k * k * (k * x1).sin(),
                1e-10,
            );
        }
    }
}

// =============================================================================
// Madelung Round-Trip
// =============================================================================

#[test]
fn madelung_round_trip() {
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;
    let mass = 1.5;
    let nu = 0.3;
    let rho_bar = 2.0;
    let amp = 0.5;

    let density = Field::scalar_field(&grid, g, |x| rho_bar + amp * (k * x[0]).sin());
    let phi_v = Field::scalar_field(&grid, g, |x| 0.1 * (k * x[0]).cos());

    let alpha = EvenField::madelung_inverse(&density, &phi_v, mass, nu);

    // Recover density
    let rho_recovered = alpha.density(mass);
    for m in [0, 4, 8, 16, 24] {
        let x = m as f64 * grid.cell_length;
        approx_eq(
            rho_recovered.at(&[m, 0, 0]).component(&[]),
            rho_bar + amp * (k * x).sin(),
            1e-12,
        );
    }

    // Recover velocity: should match ∇φ_v
    let vel = alpha.madelung_velocity(nu);
    let expected_vel = phi_v.grad();
    for m in [0, 4, 8, 16, 24] {
        let actual = vel.at(&[m, 0, 0]);
        let expected = expected_vel.at(&[m, 0, 0]);
        approx_eq(actual.component(&[1]), expected.component(&[1]), 1e-10);
        approx_eq(actual.component(&[2]), expected.component(&[2]), 1e-10);
        approx_eq(actual.component(&[3]), expected.component(&[3]), 1e-10);
    }
}

// =============================================================================
// Gradient of Plane Wave
// =============================================================================

#[test]
fn gradient_of_plane_wave() {
    // α = (cos(k x_0), sin(k x_0)) = exp(I k x_0)
    // ∇a = -k sin(k x_0) e_0, ∇b = k cos(k x_0) e_0
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let psi = EvenField::from_fn(&grid, g, |x| ((k * x[0]).cos(), (k * x[0]).sin()));
    let [grad_a, grad_b] = psi.gradient_components();

    assert_eq!(grad_a.grade(), 1);
    assert_eq!(grad_b.grade(), 1);

    for m in [0, 4, 8, 16, 24] {
        let x = m as f64 * grid.cell_length;
        let ga = grad_a.at(&[m, 0, 0]);
        let gb = grad_b.at(&[m, 0, 0]);

        approx_eq(ga.component(&[1]), -k * (k * x).sin(), 1e-10);
        approx_eq(ga.component(&[2]), 0.0, 1e-10);
        approx_eq(ga.component(&[3]), 0.0, 1e-10);

        approx_eq(gb.component(&[1]), k * (k * x).cos(), 1e-10);
        approx_eq(gb.component(&[2]), 0.0, 1e-10);
        approx_eq(gb.component(&[3]), 0.0, 1e-10);
    }
}

// =============================================================================
// Kinetic Energy of Plane Wave
// =============================================================================

#[test]
fn kinetic_energy_of_plane_wave() {
    // For α = exp(I k x_0) with |α|² = 1:
    // |∇a|² = k² sin²(kx), |∇b|² = k² cos²(kx)
    // KE density = k²/2 uniformly
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let psi = EvenField::from_fn(&grid, g, |x| ((k * x[0]).cos(), (k * x[0]).sin()));
    let ke = psi.kinetic_energy_density();

    assert_eq!(ke.grade(), 0);

    for m in [0, 4, 8, 16, 24] {
        approx_eq(ke.at(&[m, 0, 0]).component(&[]), k * k / 2.0, 1e-10);
    }
}

// =============================================================================
// Madelung Velocity of Plane Wave
// =============================================================================

#[test]
fn madelung_velocity_of_plane_wave() {
    // For α = exp(I k x_0), velocity = ν k e_0 everywhere
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;
    let nu = 0.5;

    let psi = EvenField::from_fn(&grid, g, |x| ((k * x[0]).cos(), (k * x[0]).sin()));
    let vel = psi.madelung_velocity(nu);

    assert_eq!(vel.grade(), 1);

    for m in [0, 4, 8, 16, 24] {
        let v = vel.at(&[m, 0, 0]);
        approx_eq(v.component(&[1]), nu * k, 1e-10);
        approx_eq(v.component(&[2]), 0.0, 1e-10);
        approx_eq(v.component(&[3]), 0.0, 1e-10);
    }
}

// =============================================================================
// Integrate Norm Squared
// =============================================================================

#[test]
fn integrate_norm_squared_uniform() {
    // Uniform α = (1, 0): ∫|α|² dV = V
    let g = euclidean::<3>();
    let l = 2.0;
    let grid = Grid::<3>::new(8, l);
    let volume = l.powi(3);

    let psi = EvenField::from_fn(&grid, g, |_| (1.0, 0.0));
    approx_eq(psi.integrate_norm_squared(), volume, 1e-12);

    // α = (A, 0): ∫|α|² dV = A² V
    let amp = 3.0;
    let psi2 = EvenField::from_fn(&grid, g, |_| (amp, 0.0));
    approx_eq(psi2.integrate_norm_squared(), amp * amp * volume, 1e-12);
}
