//! Spectral (FFT-based) differential operators for fields on periodic grids.
//!
//! All derivatives are computed in Fourier space: forward FFT, multiply by
//! the appropriate wavenumber factor, inverse FFT. This gives spectral
//! accuracy on periodic domains.

use ndarray::{ArrayD, IxDyn};
use ndrustfft::{Complex, FftHandler, ndfft, ndifft};

use crate::antisymmetric::n_components;
use crate::field::{Field, field_shape, spatial_indices_iter};
use crate::vector::Vector;

// =============================================================================
// Core FFT helpers
// =============================================================================

/// Forward D-dimensional complex-to-complex FFT of a real-valued array.
///
/// Input shape: [N; D] (spatial only).
/// Returns complex array of the same shape.
pub(crate) fn fft_forward<const D: usize>(
    real_data: &ArrayD<f64>,
    n: usize,
) -> ArrayD<Complex<f64>> {
    let shape: Vec<usize> = real_data.shape().to_vec();

    // Convert real to complex
    let mut complex_data = ArrayD::<Complex<f64>>::zeros(IxDyn(&shape));
    for (c, &r) in complex_data.iter_mut().zip(real_data.iter()) {
        *c = Complex::new(r, 0.0);
    }

    // FFT along each spatial axis
    let handler = FftHandler::<f64>::new(n);
    for axis in 0..D {
        let mut output = ArrayD::<Complex<f64>>::zeros(IxDyn(&shape));
        ndfft(&complex_data, &mut output, &handler, axis);
        complex_data = output;
    }

    complex_data
}

/// Inverse D-dimensional complex-to-complex FFT, returning real part.
///
/// Input shape: [N; D] (spatial only, complex).
/// Returns real array of the same shape.
pub(crate) fn fft_inverse<const D: usize>(
    complex_data: &ArrayD<Complex<f64>>,
    n: usize,
) -> ArrayD<f64> {
    let shape: Vec<usize> = complex_data.shape().to_vec();

    let mut data = complex_data.clone();
    let handler = FftHandler::<f64>::new(n);

    for axis in 0..D {
        let mut output = ArrayD::<Complex<f64>>::zeros(IxDyn(&shape));
        ndifft(&data, &mut output, &handler, axis);
        data = output;
    }

    // Extract real part
    let mut result = ArrayD::<f64>::zeros(IxDyn(&shape));
    for (r, c) in result.iter_mut().zip(data.iter()) {
        *r = c.re;
    }
    result
}

/// Extract a scalar component from a field's data array.
///
/// Given data of shape [N; D] ++ [n_components], extracts the scalar field
/// at a fixed component index, returning shape [N; D].
fn extract_component<const D: usize>(
    data: &ArrayD<f64>,
    n: usize,
    grade: usize,
    comp_idx: usize,
) -> ArrayD<f64> {
    let spatial_shape: Vec<usize> = vec![n; D];
    let mut component = ArrayD::<f64>::zeros(IxDyn(&spatial_shape));

    for spatial_idx in spatial_indices_iter::<D>(n) {
        let mut full_idx = spatial_idx.clone();
        if grade > 0 {
            full_idx.push(comp_idx);
        }
        component[IxDyn(&spatial_idx)] = data[IxDyn(&full_idx)];
    }
    component
}

/// Write a scalar component back into a field's data array.
fn write_component<const D: usize>(
    data: &mut ArrayD<f64>,
    n: usize,
    grade: usize,
    comp_idx: usize,
    component: &ArrayD<f64>,
) {
    for spatial_idx in spatial_indices_iter::<D>(n) {
        let mut full_idx = spatial_idx.clone();
        if grade > 0 {
            full_idx.push(comp_idx);
        }
        data[IxDyn(&full_idx)] = component[IxDyn(&spatial_idx)];
    }
}

// =============================================================================
// Partial Derivative
// =============================================================================

impl<const D: usize> Field<D> {
    /// Partial derivative with respect to spatial axis (physics convention).
    ///
    /// For Euclidean: `partial(1)` differentiates along x.
    /// Computed spectrally: in Fourier space, multiply by i*k_a.
    /// Grade-preserving: operates on each independent component.
    pub fn partial(&self, axis: usize) -> Field<D> {
        let internal_axis = self.metric.to_internal(axis);
        self.partial_raw(internal_axis)
    }

    /// Internal partial derivative using 0-based axis index.
    fn partial_raw(&self, axis: usize) -> Field<D> {
        assert!(
            axis < D,
            "internal axis {} out of range for {}-dimensional field",
            axis,
            D
        );
        let n = self.grid.n_cells;
        let grade = self.grade();
        let shape = field_shape::<D>(n, grade);
        let mut result_data = ArrayD::<f64>::zeros(IxDyn(&shape));

        let n_comp = if grade == 0 {
            1
        } else {
            n_components(D, grade)
        };

        for c in 0..n_comp {
            let component = extract_component::<D>(&self.data, n, grade, c);
            let mut hat = fft_forward::<D>(&component, n);

            let nyquist = n / 2;
            for freq_idx in spatial_indices_iter::<D>(n) {
                if freq_idx[axis] == nyquist {
                    hat[IxDyn(&freq_idx)] = Complex::new(0.0, 0.0);
                } else {
                    let k_a = self.grid.wavenumber(freq_idx[axis]);
                    let ik_a = Complex::new(0.0, k_a);
                    hat[IxDyn(&freq_idx)] *= ik_a;
                }
            }

            let deriv = fft_inverse::<D>(&hat, n);
            write_component::<D>(&mut result_data, n, grade, c, &deriv);
        }

        Field::new(result_data, grade, &self.grid, self.metric)
    }

    /// Gradient: raises grade by 1 via exterior derivative.
    ///
    /// grad(f) = sum_a e_a ^ d_a f
    pub fn grad(&self) -> Field<D> {
        let result_grade = self.grade() + 1;
        let mut result = Field::zeros(result_grade, &self.grid, self.metric);

        for a in 0..D {
            let df_da = self.partial_raw(a);

            // e_a as a grade-1 vector (internal construction)
            let mut e_a_data = vec![0.0; D];
            e_a_data[a] = 1.0;
            let e_a = Vector::from_sparse(e_a_data, 1, self.metric);
            let e_a_field = Field::constant(&e_a, &self.grid);

            // e_a ^ d_a f
            let term = Field::wedge(&e_a_field, &df_da);
            result = &result + &term;
        }

        result
    }

    /// Divergence: lowers grade by 1 via interior derivative.
    ///
    /// div(f) = sum_a e_a . d_a f
    pub fn div(&self) -> Field<D> {
        assert!(self.grade() >= 1, "divergence requires grade >= 1");
        let result_grade = self.grade() - 1;
        let mut result = Field::zeros(result_grade, &self.grid, self.metric);

        for a in 0..D {
            let df_da = self.partial_raw(a);

            let mut e_a_data = vec![0.0; D];
            e_a_data[a] = 1.0;
            let e_a = Vector::from_sparse(e_a_data, 1, self.metric);
            let e_a_field = Field::constant(&e_a, &self.grid);

            // e_a . d_a f (left interior product)
            let term = Field::interior_left(&e_a_field, &df_da);
            result = &result + &term;
        }

        result
    }

    /// Exterior derivative: nabla ^ f.
    ///
    /// For a vector field in 3D, this is the curl (returns bivector field).
    pub fn curl(&self) -> Field<D> {
        self.grad()
    }

    /// Laplacian: grade-preserving second derivative.
    ///
    /// Computed spectrally: multiply by -|k|^2 in Fourier space.
    pub fn laplacian(&self) -> Field<D> {
        let n = self.grid.n_cells;
        let grade = self.grade();
        let shape = field_shape::<D>(n, grade);
        let mut result_data = ArrayD::<f64>::zeros(IxDyn(&shape));

        let n_comp = if grade == 0 {
            1
        } else {
            n_components(D, grade)
        };

        for c in 0..n_comp {
            let component = extract_component::<D>(&self.data, n, grade, c);
            let mut hat = fft_forward::<D>(&component, n);

            // Multiply by -|k|^2
            let mut freq = [0usize; D];
            for freq_idx in spatial_indices_iter::<D>(n) {
                freq[..D].copy_from_slice(&freq_idx[..D]);
                let k_sq = self.grid.k_squared(&freq);
                hat[IxDyn(&freq_idx)] *= -k_sq;
            }

            let lap = fft_inverse::<D>(&hat, n);
            write_component::<D>(&mut result_data, n, grade, c, &lap);
        }

        Field::new(result_data, grade, &self.grid, self.metric)
    }

    /// Solve nabla^2 phi = f for phi on the periodic domain.
    ///
    /// Spectral method: phi_hat(k) = -f_hat(k) / |k|^2 with phi_hat(0) = 0.
    pub fn laplacian_inverse(&self) -> Field<D> {
        let n = self.grid.n_cells;
        let grade = self.grade();
        let shape = field_shape::<D>(n, grade);
        let mut result_data = ArrayD::<f64>::zeros(IxDyn(&shape));

        let n_comp = if grade == 0 {
            1
        } else {
            n_components(D, grade)
        };

        for c in 0..n_comp {
            let component = extract_component::<D>(&self.data, n, grade, c);
            let mut hat = fft_forward::<D>(&component, n);

            let mut freq = [0usize; D];
            for freq_idx in spatial_indices_iter::<D>(n) {
                freq[..D].copy_from_slice(&freq_idx[..D]);
                let k_sq = self.grid.k_squared(&freq);
                if k_sq.abs() < 1e-30 {
                    hat[IxDyn(&freq_idx)] = Complex::new(0.0, 0.0);
                } else {
                    hat[IxDyn(&freq_idx)] /= -k_sq;
                }
            }

            let solved = fft_inverse::<D>(&hat, n);
            write_component::<D>(&mut result_data, n, grade, c, &solved);
        }

        Field::new(result_data, grade, &self.grid, self.metric)
    }
}
