use std::f64::consts::PI;

use morphis::field::Field;
use morphis::grid::Grid;
use morphis::metric::euclidean;
use morphis::vector::Vector;
use ndarray::{ArrayD, IxDyn};

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
// Field Construction and Access
// =============================================================================

#[test]
fn scalar_field_from_fn() {
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(8, 1.0);
    let f = Field::scalar_field(&grid, g, |x| x[0] + x[1]);

    let val = f.at(&[2, 3, 0]);
    assert_eq!(val.grade(), 0);
    let expected = 2.0 * grid.cell_length + 3.0 * grid.cell_length;
    approx_eq(val.component(&[]), expected, 1e-12);
}

#[test]
fn constant_field_roundtrip() {
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(4, 10.0);

    let mut v_data = ArrayD::<f64>::zeros(IxDyn(&[3]));
    v_data[IxDyn(&[0])] = 1.0;
    v_data[IxDyn(&[1])] = 2.0;
    v_data[IxDyn(&[2])] = 3.0;
    let v = Vector::new(v_data, 1, g);

    let f = Field::constant(&v, &grid);
    assert_eq!(f.grade(), 1);

    // Check at arbitrary point
    let extracted = f.at(&[1, 2, 3]);
    approx_eq(extracted.component(&[1]), 1.0, 1e-12);
    approx_eq(extracted.component(&[2]), 2.0, 1e-12);
    approx_eq(extracted.component(&[3]), 3.0, 1e-12);
}

#[test]
fn set_and_get_roundtrip() {
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(4, 10.0);
    let mut f = Field::zeros(1, &grid, g);

    let mut v_data = ArrayD::<f64>::zeros(IxDyn(&[3]));
    v_data[IxDyn(&[0])] = 5.0;
    v_data[IxDyn(&[2])] = -3.0;
    let v = Vector::new(v_data, 1, g);

    f.set(&[1, 2, 3], &v);
    let extracted = f.at(&[1, 2, 3]);
    approx_eq(extracted.component(&[1]), 5.0, 1e-12);
    approx_eq(extracted.component(&[3]), -3.0, 1e-12);
}

#[test]
fn integration_of_constant() {
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(8, 10.0);
    let f = Field::scalar_field(&grid, g, |_| 2.0);

    let volume = 10.0_f64.powi(3);
    approx_eq(f.integrate(), 2.0 * volume, 1e-10);
}

// =============================================================================
// Pointwise Algebra
// =============================================================================

#[test]
fn field_addition() {
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(4, 1.0);
    let f1 = Field::scalar_field(&grid, g, |x| x[0]);
    let f2 = Field::scalar_field(&grid, g, |x| x[1]);
    let sum = &f1 + &f2;

    let val = sum.at(&[2, 3, 0]);
    let expected = 2.0 * grid.cell_length + 3.0 * grid.cell_length;
    approx_eq(val.component(&[]), expected, 1e-12);
}

#[test]
fn field_scalar_multiply() {
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(4, 1.0);
    let f = Field::scalar_field(&grid, g, |x| x[0]);
    let scaled = &f * 3.0;

    let val = scaled.at(&[2, 0, 0]);
    approx_eq(val.component(&[]), 3.0 * 2.0 * grid.cell_length, 1e-12);
}

#[test]
fn field_norm_squared_of_basis() {
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(4, 1.0);

    // Constant unit vector field along e1
    let mut v_data = ArrayD::<f64>::zeros(IxDyn(&[3]));
    v_data[IxDyn(&[0])] = 1.0;
    let v = Vector::new(v_data, 1, g);
    let f = Field::constant(&v, &grid);

    let norm_sq = f.norm_squared();
    assert_eq!(norm_sq.grade(), 0);

    // Should be 1.0 everywhere
    let val = norm_sq.at(&[1, 2, 3]);
    approx_eq(val.component(&[]), 1.0, 1e-12);
}

// =============================================================================
// Spectral Derivatives
// =============================================================================

#[test]
fn gradient_of_sin() {
    // grad(sin(kx)) = k*cos(kx) e1
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let f = Field::scalar_field(&grid, g, |x| (k * x[0]).sin());
    let grad_f = f.grad();
    assert_eq!(grad_f.grade(), 1);

    // Check at several points
    for m in [0, 4, 8, 16, 24] {
        let v = grad_f.at(&[m, 0, 0]);
        let x = m as f64 * grid.cell_length;
        let expected = k * (k * x).cos();
        approx_eq(v.component(&[1]), expected, 1e-10);
        approx_eq(v.component(&[2]), 0.0, 1e-10);
        approx_eq(v.component(&[3]), 0.0, 1e-10);
    }
}

#[test]
fn divergence_of_gradient_equals_laplacian() {
    // div(grad(f)) == laplacian(f) for any scalar field
    let g = euclidean::<3>();
    let n = 16;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let f = Field::scalar_field(&grid, g, |x| (k * x[0]).sin() * (k * x[1]).cos());

    let div_grad = f.grad().div();
    let lap = f.laplacian();

    // Compare pointwise
    for m in [0, 4, 8, 12] {
        for p in [0, 4, 8, 12] {
            let dg = div_grad.at(&[m, p, 0]);
            let lp = lap.at(&[m, p, 0]);
            approx_eq(dg.component(&[]), lp.component(&[]), 1e-10);
        }
    }
}

#[test]
fn curl_of_gradient_is_zero() {
    // ∇ ∧ (∇f) = 0 for any scalar field
    let g = euclidean::<3>();
    let n = 16;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let f = Field::scalar_field(&grid, g, |x| {
        (k * x[0]).sin() * (k * x[1]).cos() * (k * x[2]).sin()
    });

    let curl_grad = f.grad().curl();
    assert_eq!(curl_grad.grade(), 2);
    assert!(curl_grad.is_zero(1e-10));
}

#[test]
fn exterior_derivative_squared_is_zero() {
    // d²=0: (∇∧)(∇∧)v = 0 for any vector field
    // curl of curl: grade 1 → grade 2 → grade 3
    let g = euclidean::<3>();
    let n = 16;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let v = Field::from_fn(1, &grid, g, |x| {
        let mut data = ArrayD::<f64>::zeros(IxDyn(&[3]));
        data[IxDyn(&[0])] = (k * x[1]).sin();
        data[IxDyn(&[1])] = (k * x[2]).cos();
        data[IxDyn(&[2])] = (k * x[0]).sin();
        Vector::new(data, 1, g)
    });

    // ∇∧v: grade 1 → grade 2
    let curl_v = v.curl();
    assert_eq!(curl_v.grade(), 2);

    // ∇∧(∇∧v): grade 2 → grade 3 (should be zero)
    let curl_curl = curl_v.curl();
    assert_eq!(curl_curl.grade(), 3);
    assert!(curl_curl.is_zero(1e-10));
}

#[test]
fn laplacian_of_sin() {
    // ∇²sin(kx) = -k²sin(kx)
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let f = Field::scalar_field(&grid, g, |x| (k * x[0]).sin());
    let lap = f.laplacian();

    for m in [0, 4, 8, 16, 24] {
        let val = lap.at(&[m, 0, 0]);
        let x = m as f64 * grid.cell_length;
        let expected = -k * k * (k * x).sin();
        approx_eq(val.component(&[]), expected, 1e-10);
    }
}

#[test]
fn laplacian_inverse_roundtrip() {
    // laplacian(laplacian_inverse(f)) = f for zero-mean f
    let g = euclidean::<3>();
    let n = 16;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    // sin is already zero-mean on [0, L)
    let f = Field::scalar_field(&grid, g, |x| (k * x[0]).sin());
    let roundtrip = f.laplacian_inverse().laplacian();

    for m in [0, 4, 8, 12] {
        let original = f.at(&[m, 0, 0]);
        let recovered = roundtrip.at(&[m, 0, 0]);
        approx_eq(original.component(&[]), recovered.component(&[]), 1e-10);
    }
}

#[test]
fn laplacian_inverse_known_solution() {
    // laplacian_inverse(sin(kx)) = -sin(kx)/k²
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let f = Field::scalar_field(&grid, g, |x| (k * x[0]).sin());
    let phi = f.laplacian_inverse();

    for m in [0, 4, 8, 16, 24] {
        let val = phi.at(&[m, 0, 0]);
        let x = m as f64 * grid.cell_length;
        let expected = -(k * x).sin() / (k * k);
        approx_eq(val.component(&[]), expected, 1e-10);
    }
}

#[test]
fn laplacian_inverse_of_constant_is_zero() {
    // Zero mode is projected out
    let g = euclidean::<3>();
    let n = 8;
    let grid = Grid::<3>::new(n, 1.0);

    let f = Field::scalar_field(&grid, g, |_| 5.0);
    let phi = f.laplacian_inverse();
    assert!(phi.is_zero(1e-12));
}

#[test]
fn grade_propagation() {
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(8, 1.0);

    let scalar = Field::zeros(0, &grid, g);
    let vector = Field::zeros(1, &grid, g);
    let bivector = Field::zeros(2, &grid, g);

    // Grad raises
    assert_eq!(scalar.grad().grade(), 1);
    assert_eq!(vector.grad().grade(), 2);

    // Div lowers
    assert_eq!(vector.div().grade(), 0);
    assert_eq!(bivector.div().grade(), 1);

    // Laplacian preserves
    assert_eq!(scalar.laplacian().grade(), 0);
    assert_eq!(vector.laplacian().grade(), 1);
}

// =============================================================================
// Spectral Derivatives on Vector Fields
// =============================================================================

#[test]
fn laplacian_of_vector_field() {
    // v = sin(k x_0) e_0 → ∇²v = -k² sin(k x_0) e_0
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let v = Field::from_fn(1, &grid, g, |x| {
        let mut data = ArrayD::<f64>::zeros(IxDyn(&[3]));
        data[IxDyn(&[0])] = (k * x[0]).sin();
        Vector::new(data, 1, g)
    });

    let lap = v.laplacian();
    assert_eq!(lap.grade(), 1);

    for m in [0, 4, 8, 16, 24] {
        let val = lap.at(&[m, 0, 0]);
        let x = m as f64 * grid.cell_length;
        approx_eq(val.component(&[1]), -k * k * (k * x).sin(), 1e-10);
        approx_eq(val.component(&[2]), 0.0, 1e-10);
        approx_eq(val.component(&[3]), 0.0, 1e-10);
    }
}

#[test]
fn laplacian_inverse_of_vector_field() {
    // laplacian_inverse(sin(k x_0) e_0) = -sin(k x_0)/k² e_0
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let v = Field::from_fn(1, &grid, g, |x| {
        let mut data = ArrayD::<f64>::zeros(IxDyn(&[3]));
        data[IxDyn(&[0])] = (k * x[0]).sin();
        Vector::new(data, 1, g)
    });

    let phi = v.laplacian_inverse();
    assert_eq!(phi.grade(), 1);

    for m in [0, 4, 8, 16, 24] {
        let val = phi.at(&[m, 0, 0]);
        let x = m as f64 * grid.cell_length;
        approx_eq(val.component(&[1]), -(k * x).sin() / (k * k), 1e-10);
        approx_eq(val.component(&[2]), 0.0, 1e-10);
        approx_eq(val.component(&[3]), 0.0, 1e-10);
    }
}

// =============================================================================
// Partial Derivative Directly
// =============================================================================

#[test]
fn partial_of_sin() {
    // ∂_0 sin(k x_0) = k cos(k x_0)
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let f = Field::scalar_field(&grid, g, |x| (k * x[0]).sin());
    let df = f.partial(1); // x-direction = physics index 1

    for m in [0, 4, 8, 16, 24] {
        let x = m as f64 * grid.cell_length;
        approx_eq(df.at(&[m, 0, 0]).component(&[]), k * (k * x).cos(), 1e-10);
    }
}

#[test]
fn partial_orthogonal_axis_is_zero() {
    // d_2 sin(k x_1) = 0
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let f = Field::scalar_field(&grid, g, |x| (k * x[0]).sin());
    let df = f.partial(2); // y-direction = physics index 2

    assert!(df.is_zero(1e-10));
}

// =============================================================================
// Higher Harmonics
// =============================================================================

#[test]
fn laplacian_higher_harmonics() {
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);

    for mode in [2, 3] {
        let k = 2.0 * PI * mode as f64 / l;
        let f = Field::scalar_field(&grid, g, |x| (k * x[0]).sin());
        let lap = f.laplacian();

        for m in [0, 4, 8, 16] {
            let x = m as f64 * grid.cell_length;
            approx_eq(
                lap.at(&[m, 0, 0]).component(&[]),
                -k * k * (k * x).sin(),
                1e-10,
            );
        }
    }
}

#[test]
fn gradient_higher_harmonics() {
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);

    for mode in [2, 3] {
        let k = 2.0 * PI * mode as f64 / l;
        let f = Field::scalar_field(&grid, g, |x| (k * x[0]).sin());
        let grad_f = f.grad();

        for m in [0, 4, 8, 16] {
            let val = grad_f.at(&[m, 0, 0]);
            let x = m as f64 * grid.cell_length;
            approx_eq(val.component(&[1]), k * (k * x).cos(), 1e-10);
            approx_eq(val.component(&[2]), 0.0, 1e-10);
        }
    }
}

// =============================================================================
// Multi-Dimensional Modes
// =============================================================================

#[test]
fn laplacian_multi_dimensional() {
    // ∇²[sin(kx) sin(ky)] = -2k² sin(kx) sin(ky)
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let f = Field::scalar_field(&grid, g, |x| (k * x[0]).sin() * (k * x[1]).sin());
    let lap = f.laplacian();

    for m in [0, 4, 8, 16] {
        for p in [0, 4, 8, 16] {
            let x0 = m as f64 * grid.cell_length;
            let x1 = p as f64 * grid.cell_length;
            let expected = -2.0 * k * k * (k * x0).sin() * (k * x1).sin();
            approx_eq(lap.at(&[m, p, 0]).component(&[]), expected, 1e-10);
        }
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn integration_of_sin_squared() {
    // ∫ sin²(kx) dV = V / 2
    let g = euclidean::<3>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let f = Field::scalar_field(&grid, g, |x| {
        let s = (k * x[0]).sin();
        s * s
    });

    let volume = l.powi(3);
    approx_eq(f.integrate(), volume / 2.0, 1e-12);
}

#[test]
fn integrate_norm_squared_constant_vector() {
    // Constant unit vector field: ∫ |v|² dV = V
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(8, 10.0);

    let mut v_data = ArrayD::<f64>::zeros(IxDyn(&[3]));
    v_data[IxDyn(&[0])] = 1.0;
    let v = Vector::new(v_data, 1, g);
    let f = Field::constant(&v, &grid);

    let volume = 10.0_f64.powi(3);
    approx_eq(f.integrate_norm_squared(), volume, 1e-10);
}

// =============================================================================
// Component Field Extraction
// =============================================================================

#[test]
fn component_field_extraction() {
    let g = euclidean::<3>();
    let n = 8;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);

    let v = Field::from_fn(1, &grid, g, |x| {
        let mut data = ArrayD::<f64>::zeros(IxDyn(&[3]));
        data[IxDyn(&[0])] = x[0];
        data[IxDyn(&[1])] = x[1] * 2.0;
        Vector::new(data, 1, g)
    });

    let c0 = v.component_field(&[1]); // x-component (physics index 1)
    let c1 = v.component_field(&[2]); // y-component (physics index 2)

    assert_eq!(c0.grade(), 0);
    assert_eq!(c1.grade(), 0);

    for m in [0, 2, 4, 6] {
        let x0 = m as f64 * grid.cell_length;
        approx_eq(c0.at(&[m, 0, 0]).component(&[]), x0, 1e-12);
    }
    for p in [0, 2, 4, 6] {
        let x1 = p as f64 * grid.cell_length;
        approx_eq(c1.at(&[0, p, 0]).component(&[]), x1 * 2.0, 1e-12);
    }
}

// =============================================================================
// Field Products
// =============================================================================

#[test]
fn field_wedge_product() {
    // e_0 ∧ e_1 = e_01 bivector
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(4, 1.0);

    let mut e0_data = ArrayD::<f64>::zeros(IxDyn(&[3]));
    e0_data[IxDyn(&[0])] = 1.0;
    let e0 = Vector::new(e0_data, 1, g);
    let f0 = Field::constant(&e0, &grid);

    let mut e1_data = ArrayD::<f64>::zeros(IxDyn(&[3]));
    e1_data[IxDyn(&[1])] = 1.0;
    let e1 = Vector::new(e1_data, 1, g);
    let f1 = Field::constant(&e1, &grid);

    let w = Field::wedge(&f0, &f1);
    assert_eq!(w.grade(), 2);

    let val = w.at(&[0, 0, 0]);
    approx_eq(val.component(&[1, 2]), 1.0, 1e-12);
    approx_eq(val.component(&[2, 1]), -1.0, 1e-12);
    approx_eq(val.component(&[1, 1]), 0.0, 1e-12);
}

#[test]
fn field_scalar_product() {
    // <e_0, e_0> = 1, <e_0, e_1> = 0 in Euclidean metric
    let g = euclidean::<3>();
    let grid = Grid::<3>::new(4, 1.0);

    let mut e0_data = ArrayD::<f64>::zeros(IxDyn(&[3]));
    e0_data[IxDyn(&[0])] = 1.0;
    let e0 = Vector::new(e0_data, 1, g);
    let f0 = Field::constant(&e0, &grid);

    let sp = Field::scalar_product(&f0, &f0);
    assert_eq!(sp.grade(), 0);
    approx_eq(sp.at(&[0, 0, 0]).component(&[]), 1.0, 1e-12);
}

// =============================================================================
// Pointwise Scale
// =============================================================================

#[test]
fn pointwise_scale_vector_field() {
    let g = euclidean::<3>();
    let n = 8;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);

    let mut e0_data = ArrayD::<f64>::zeros(IxDyn(&[3]));
    e0_data[IxDyn(&[0])] = 1.0;
    let e0 = Vector::new(e0_data, 1, g);
    let v = Field::constant(&e0, &grid);

    let s = Field::scalar_field(&grid, g, |x| x[0]);
    let scaled = Field::pointwise_scale(&s, &v);

    assert_eq!(scaled.grade(), 1);
    for m in [0, 2, 4, 6] {
        let x = m as f64 * grid.cell_length;
        approx_eq(scaled.at(&[m, 0, 0]).component(&[1]), x, 1e-12);
        approx_eq(scaled.at(&[m, 0, 0]).component(&[2]), 0.0, 1e-12);
    }
}

// =============================================================================
// 2D Fields
// =============================================================================

#[test]
fn laplacian_of_sin_2d() {
    // ∇²sin(kx) = -k²sin(kx) on a 2D grid
    let g = euclidean::<2>();
    let n = 32;
    let l = 1.0;
    let grid = Grid::<2>::new(n, l);
    let k = 2.0 * PI / l;

    let f = Field::scalar_field(&grid, g, |x| (k * x[0]).sin());
    let lap = f.laplacian();

    for m in [0, 4, 8, 16, 24] {
        let x = m as f64 * grid.cell_length;
        approx_eq(
            lap.at(&[m, 0]).component(&[]),
            -k * k * (k * x).sin(),
            1e-10,
        );
    }
}

// =============================================================================
// Laplacian Self-Adjointness
// =============================================================================

#[test]
fn laplacian_self_adjoint() {
    // <f | ∇²g> = <∇²f | g>
    let g = euclidean::<3>();
    let n = 16;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let f = Field::scalar_field(&grid, g, |x| (k * x[0]).sin());
    let h = Field::scalar_field(&grid, g, |x| (2.0 * k * x[1]).cos());

    let lap_f = f.laplacian();
    let lap_h = h.laplacian();

    let lhs = Field::scalar_product(&f, &lap_h).integrate();
    let rhs = Field::scalar_product(&lap_f, &h).integrate();
    approx_eq(lhs, rhs, 1e-10);
}

#[test]
fn laplacian_negative_semi_definite() {
    // -<f | ∇²f> >= 0
    let g = euclidean::<3>();
    let n = 16;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);
    let k = 2.0 * PI / l;

    let f = Field::scalar_field(&grid, g, |x| {
        (k * x[0]).sin() + 0.5 * (2.0 * k * x[1]).cos()
    });

    let lap_f = f.laplacian();
    let inner = Field::scalar_product(&f, &lap_f).integrate();
    assert!(
        -inner >= -1e-12,
        "-<f|∇²f> should be non-negative, got {}",
        -inner
    );
}

// =============================================================================
// Nyquist Mode in Odd-Order Derivatives
// =============================================================================

#[test]
fn nyquist_mode_zeroed_in_partial() {
    // A field with energy only at the Nyquist mode should produce
    // zero output from partial (odd-order derivative)
    let g = euclidean::<3>();
    let n = 8;
    let l = 1.0;
    let grid = Grid::<3>::new(n, l);

    // Nyquist mode: m = N/2 = 4, k = 2π·4/L
    let k_nyq = 2.0 * PI * (n as f64 / 2.0) / l;
    let f = Field::scalar_field(&grid, g, |x| (k_nyq * x[0]).cos());

    let df = f.partial(1); // x-direction = physics index 1
    assert!(
        df.is_zero(1e-10),
        "Nyquist mode should be zeroed in odd-order derivative"
    );
}
