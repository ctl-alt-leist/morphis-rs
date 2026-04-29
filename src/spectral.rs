//! Spectral (FFT-based) differential operators for fields on periodic grids.
//!
//! All derivatives are computed in Fourier space: forward FFT, multiply by
//! the appropriate wavenumber factor, inverse FFT. This gives spectral
//! accuracy on periodic domains.

use ndarray::{ArrayD, IxDyn};
use ndrustfft::{Complex, FftHandler, ndfft, ndifft};

use crate::field::{Field, field_shape, spatial_indices_iter, tensor_indices_iter};
use crate::vector::Vector;

// =============================================================================
// Core FFT helpers
// =============================================================================

/// Forward D-dimensional complex-to-complex FFT of a real-valued array.
///
/// Input shape: [N; D] (spatial only).
/// Returns complex array of the same shape.
fn fft_forward<const D: usize>(real_data: &ArrayD<f64>, n: usize) -> ArrayD<Complex<f64>> {
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
fn fft_inverse<const D: usize>(complex_data: &ArrayD<Complex<f64>>, n: usize) -> ArrayD<f64> {
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
/// Given data of shape [N; D] ++ [D; grade], extracts the scalar field
/// at a fixed tensor index, returning shape [N; D].
fn extract_component<const D: usize>(
    data: &ArrayD<f64>,
    n: usize,
    grade: usize,
    tensor_idx: &[usize],
) -> ArrayD<f64> {
    let spatial_shape: Vec<usize> = vec![n; D];
    let mut component = ArrayD::<f64>::zeros(IxDyn(&spatial_shape));

    for spatial_idx in spatial_indices_iter::<D>(n) {
        let mut full_idx = spatial_idx.clone();
        if grade > 0 {
            full_idx.extend_from_slice(tensor_idx);
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
    tensor_idx: &[usize],
    component: &ArrayD<f64>,
) {
    for spatial_idx in spatial_indices_iter::<D>(n) {
        let mut full_idx = spatial_idx.clone();
        if grade > 0 {
            full_idx.extend_from_slice(tensor_idx);
        }
        data[IxDyn(&full_idx)] = component[IxDyn(&spatial_idx)];
    }
}

// =============================================================================
// Partial Derivative
// =============================================================================

impl<const D: usize> Field<D> {
    /// Partial derivative with respect to spatial axis `a`.
    ///
    /// Computed spectrally: in Fourier space, multiply by i*k_a.
    /// Grade-preserving: operates on each tensor component independently.
    pub fn partial(&self, axis: usize) -> Field<D> {
        assert!(
            axis < D,
            "axis {} out of range for {}-dimensional field",
            axis,
            D
        );
        let n = self.grid.n_cells;
        let grade = self.grade();
        let shape = field_shape::<D>(n, grade);
        let mut result_data = ArrayD::<f64>::zeros(IxDyn(&shape));

        let tensor_indices: Vec<Vec<usize>> = if grade == 0 {
            vec![vec![]]
        } else {
            tensor_indices_iter(D, grade)
        };

        for tensor_idx in &tensor_indices {
            // Extract this scalar component
            let component = extract_component::<D>(&self.data, n, grade, tensor_idx);

            // Forward FFT
            let mut hat = fft_forward::<D>(&component, n);

            // Multiply by i*k_a at each frequency
            for freq_idx in spatial_indices_iter::<D>(n) {
                let k_a = self.grid.wavenumber(freq_idx[axis]);
                let ik_a = Complex::new(0.0, k_a);
                hat[IxDyn(&freq_idx)] *= ik_a;
            }

            // Inverse FFT
            let deriv = fft_inverse::<D>(&hat, n);

            // Write back
            write_component::<D>(&mut result_data, n, grade, tensor_idx, &deriv);
        }

        Field::new(result_data, grade, &self.grid, self.metric)
    }

    /// Gradient: raises grade by 1 via exterior derivative.
    ///
    /// grad(f) = Σ_a e_a ∧ ∂_a f
    ///
    /// Scalar field → vector field, vector field → bivector field, etc.
    pub fn grad(&self) -> Field<D> {
        let result_grade = self.grade() + 1;
        let mut result = Field::zeros(result_grade, &self.grid, self.metric);

        for a in 0..D {
            let df_da = self.partial(a);

            // e_a as a grade-1 vector
            let mut e_a_data = ArrayD::<f64>::zeros(IxDyn(&[D]));
            e_a_data[IxDyn(&[a])] = 1.0;
            let e_a = Vector::new(e_a_data, 1, self.metric);
            let e_a_field = Field::constant(&e_a, &self.grid);

            // e_a ∧ ∂_a f
            let term = Field::wedge(&e_a_field, &df_da);
            result = &result + &term;
        }

        result
    }

    /// Divergence: lowers grade by 1 via interior derivative.
    ///
    /// div(f) = Σ_a e_a ⌋ ∂_a f
    ///
    /// Vector field → scalar field, bivector field → vector field, etc.
    pub fn div(&self) -> Field<D> {
        assert!(self.grade() >= 1, "divergence requires grade >= 1");
        let result_grade = self.grade() - 1;
        let mut result = Field::zeros(result_grade, &self.grid, self.metric);

        for a in 0..D {
            let df_da = self.partial(a);

            // e_a as a grade-1 vector
            let mut e_a_data = ArrayD::<f64>::zeros(IxDyn(&[D]));
            e_a_data[IxDyn(&[a])] = 1.0;
            let e_a = Vector::new(e_a_data, 1, self.metric);
            let e_a_field = Field::constant(&e_a, &self.grid);

            // e_a ⌋ ∂_a f (left interior product)
            let term = Field::interior_left(&e_a_field, &df_da);
            result = &result + &term;
        }

        result
    }

    /// Exterior derivative: ∇ ∧ f.
    ///
    /// For a vector field in 3D, this is the curl (returns bivector field).
    /// Grade of result = grade(self) + 1.
    pub fn curl(&self) -> Field<D> {
        // The curl is the same as grad for the exterior derivative:
        // ∇ ∧ f = Σ_a e_a ∧ ∂_a f
        // This is identical to grad() — both are the exterior derivative.
        self.grad()
    }

    /// Laplacian: grade-preserving second derivative.
    ///
    /// ∇²f = Σ_a ∂²_a f
    ///
    /// Computed spectrally: multiply by -|k|² in Fourier space.
    pub fn laplacian(&self) -> Field<D> {
        let n = self.grid.n_cells;
        let grade = self.grade();
        let shape = field_shape::<D>(n, grade);
        let mut result_data = ArrayD::<f64>::zeros(IxDyn(&shape));

        let tensor_indices: Vec<Vec<usize>> = if grade == 0 {
            vec![vec![]]
        } else {
            tensor_indices_iter(D, grade)
        };

        for tensor_idx in &tensor_indices {
            let component = extract_component::<D>(&self.data, n, grade, tensor_idx);
            let mut hat = fft_forward::<D>(&component, n);

            // Multiply by -|k|^2
            let mut freq = [0usize; D];
            for freq_idx in spatial_indices_iter::<D>(n) {
                freq[..D].copy_from_slice(&freq_idx[..D]);
                let k_sq = self.grid.k_squared(&freq);
                hat[IxDyn(&freq_idx)] *= -k_sq;
            }

            let lap = fft_inverse::<D>(&hat, n);
            write_component::<D>(&mut result_data, n, grade, tensor_idx, &lap);
        }

        Field::new(result_data, grade, &self.grid, self.metric)
    }

    /// Solve ∇²φ = f for φ on the periodic domain.
    ///
    /// Spectral method: φ_hat(k) = -f_hat(k) / |k|² with φ_hat(0) = 0.
    /// The zero mode is projected out (returns the zero-mean solution).
    /// Grade-preserving: operates on each component independently.
    pub fn laplacian_inverse(&self) -> Field<D> {
        let n = self.grid.n_cells;
        let grade = self.grade();
        let shape = field_shape::<D>(n, grade);
        let mut result_data = ArrayD::<f64>::zeros(IxDyn(&shape));

        let tensor_indices: Vec<Vec<usize>> = if grade == 0 {
            vec![vec![]]
        } else {
            tensor_indices_iter(D, grade)
        };

        for tensor_idx in &tensor_indices {
            let component = extract_component::<D>(&self.data, n, grade, tensor_idx);
            let mut hat = fft_forward::<D>(&component, n);

            let mut freq = [0usize; D];
            for freq_idx in spatial_indices_iter::<D>(n) {
                freq[..D].copy_from_slice(&freq_idx[..D]);
                let k_sq = self.grid.k_squared(&freq);
                if k_sq.abs() < 1e-30 {
                    // Zero mode: set to zero (no unique mean for periodic Poisson)
                    hat[IxDyn(&freq_idx)] = Complex::new(0.0, 0.0);
                } else {
                    hat[IxDyn(&freq_idx)] /= -k_sq;
                }
            }

            let solved = fft_inverse::<D>(&hat, n);
            write_component::<D>(&mut result_data, n, grade, tensor_idx, &solved);
        }

        Field::new(result_data, grade, &self.grid, self.metric)
    }
}
