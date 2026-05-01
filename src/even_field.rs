//! Even subalgebra field: scalar + pseudoscalar values on a periodic grid.
//!
//! In 3D, the even subalgebra G^+ = G^0 ⊕ G^D is isomorphic to the complex
//! numbers via the pseudoscalar I = e1 ∧ e2 ∧ e3, which satisfies I² = -1.
//! Each grid point holds α = a + bI where a is the scalar part and b is the
//! pseudoscalar coefficient.

use ndarray::{ArrayD, IxDyn};
use ndrustfft::Complex;

use crate::field::{Field, field_shape, spatial_indices_iter};
use crate::grid::Grid;
use crate::metric::Metric;
use crate::multivector::MultiVector;
use crate::spectral::{fft_forward, fft_inverse};
use crate::vector::Vector;

/// A field valued in the even subalgebra G^+ = G^0 ⊕ G^D.
///
/// In 3D, this is isomorphic to a complex-valued field: each point
/// holds a + bI where I is the unit pseudoscalar.
///
/// This type is specialized for odd-dimensional spaces where I² = -1.
pub struct EvenField<const D: usize> {
    /// Scalar (grade-0) part: shape [N; D].
    pub scalar: ArrayD<f64>,
    /// Pseudoscalar coefficient (grade-D part, without the I factor): shape [N; D].
    pub pseudoscalar: ArrayD<f64>,
    /// Metric defining the inner product structure.
    pub metric: Metric<D>,
    /// Grid geometry.
    pub grid: Grid<D>,
}

impl<const D: usize> EvenField<D> {
    /// Create a zero even field.
    pub fn zeros(grid: &Grid<D>, metric: Metric<D>) -> Self {
        let shape: Vec<usize> = vec![grid.n_cells; D];
        Self {
            scalar: ArrayD::zeros(IxDyn(&shape)),
            pseudoscalar: ArrayD::zeros(IxDyn(&shape)),
            metric,
            grid: *grid,
        }
    }

    /// Create from scalar and pseudoscalar arrays.
    pub fn new(
        scalar: ArrayD<f64>,
        pseudoscalar: ArrayD<f64>,
        grid: &Grid<D>,
        metric: Metric<D>,
    ) -> Self {
        let expected_shape: Vec<usize> = vec![grid.n_cells; D];
        assert_eq!(scalar.shape(), expected_shape.as_slice());
        assert_eq!(pseudoscalar.shape(), expected_shape.as_slice());
        Self {
            scalar,
            pseudoscalar,
            metric,
            grid: *grid,
        }
    }

    /// Create from a function returning (scalar_part, pseudoscalar_coeff).
    pub fn from_fn(grid: &Grid<D>, metric: Metric<D>, f: impl Fn(&[f64; D]) -> (f64, f64)) -> Self {
        let n = grid.n_cells;
        let shape: Vec<usize> = vec![n; D];
        let mut scalar = ArrayD::zeros(IxDyn(&shape));
        let mut pseudoscalar = ArrayD::zeros(IxDyn(&shape));

        let mut indices = [0usize; D];
        for spatial_idx in spatial_indices_iter::<D>(n) {
            indices[..D].copy_from_slice(&spatial_idx[..D]);
            let pos = grid.position(&indices);
            let (s, p) = f(&pos);
            scalar[IxDyn(&spatial_idx)] = s;
            pseudoscalar[IxDyn(&spatial_idx)] = p;
        }

        Self {
            scalar,
            pseudoscalar,
            metric,
            grid: *grid,
        }
    }

    /// Reversal (complex conjugation): (a + bI) → (a - bI).
    pub fn rev(&self) -> Self {
        Self {
            scalar: self.scalar.clone(),
            pseudoscalar: -&self.pseudoscalar,
            metric: self.metric,
            grid: self.grid,
        }
    }

    /// Pointwise product: (a + bI)(c + dI) = (ac - bd) + (ad + bc)I.
    ///
    /// Closed in the even subalgebra (uses I² = -1 for odd D).
    pub fn mul(&self, other: &EvenField<D>) -> EvenField<D> {
        // I^2 = -1 in odd dimensions (D = 1, 3, 5, ...)
        // For even D, I^2 = +1 — this type assumes odd D.
        let real = &self.scalar * &other.scalar - &self.pseudoscalar * &other.pseudoscalar;
        let imag = &self.scalar * &other.pseudoscalar + &self.pseudoscalar * &other.scalar;
        Self {
            scalar: real,
            pseudoscalar: imag,
            metric: self.metric,
            grid: self.grid,
        }
    }

    /// Norm squared: α · α_rev = a² + b² (returns a scalar field).
    pub fn norm_squared(&self) -> Field<D> {
        let data = &self.scalar * &self.scalar + &self.pseudoscalar * &self.pseudoscalar;
        Field::new(data, 0, &self.grid, self.metric)
    }

    /// Phase rotation: multiply by exp(Iθ) = cos(θ) + sin(θ)·I pointwise.
    ///
    /// The angle field must be a scalar field (grade 0) on the same grid.
    /// This is the kinetic-step operation in split-step integration.
    pub fn rotate_phase(&self, angle: &Field<D>) -> EvenField<D> {
        assert_eq!(angle.grade(), 0, "angle must be a scalar field");
        assert_eq!(angle.grid, self.grid, "grids must match");

        let n = self.grid.n_cells;
        let shape: Vec<usize> = vec![n; D];
        let mut result_scalar = ArrayD::zeros(IxDyn(&shape));
        let mut result_pseudo = ArrayD::zeros(IxDyn(&shape));

        for spatial_idx in spatial_indices_iter::<D>(n) {
            let theta = angle.data[IxDyn(&spatial_idx)];
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let a = self.scalar[IxDyn(&spatial_idx)];
            let b = self.pseudoscalar[IxDyn(&spatial_idx)];

            // (a + bI) * (cos θ + sin θ I) = (a cos θ - b sin θ) + (a sin θ + b cos θ) I
            result_scalar[IxDyn(&spatial_idx)] = a * cos_t - b * sin_t;
            result_pseudo[IxDyn(&spatial_idx)] = a * sin_t + b * cos_t;
        }

        Self {
            scalar: result_scalar,
            pseudoscalar: result_pseudo,
            metric: self.metric,
            grid: self.grid,
        }
    }

    /// Extract density: ρ = m · |α|² = m · (a² + b²).
    pub fn density(&self, mass: f64) -> Field<D> {
        let norm_sq = &self.scalar * &self.scalar + &self.pseudoscalar * &self.pseudoscalar;
        let data = &norm_sq * mass;
        Field::new(data, 0, &self.grid, self.metric)
    }

    /// Pointwise extraction as MultiVector<D>.
    pub fn at(&self, indices: &[usize]) -> MultiVector<D> {
        assert_eq!(indices.len(), D, "expected {} spatial indices", D);
        let a = self.scalar[IxDyn(indices)];
        let b = self.pseudoscalar[IxDyn(indices)];

        let scalar_vec = Vector::scalar(a, self.metric);

        // Pseudoscalar: grade-D, the fully antisymmetric component
        let pseudo_shape: Vec<usize> = vec![D; D];
        let mut pseudo_data = ArrayD::zeros(IxDyn(&pseudo_shape));
        // The pseudoscalar has one independent component: e1∧e2∧...∧eD
        // In full tensor storage, this corresponds to the Levi-Civita pattern
        set_pseudoscalar_component::<D>(&mut pseudo_data, b);
        let pseudo_vec = Vector::new(pseudo_data, D, self.metric);

        let mut mv = MultiVector::from_vector(scalar_vec);
        if b.abs() > 1e-15 {
            mv = &mv + &MultiVector::from_vector(pseudo_vec);
        }
        mv
    }

    /// Grid geometry.
    pub fn grid(&self) -> &Grid<D> {
        &self.grid
    }

    /// Integrated norm squared: ∫ |α|² dV = Σ (a² + b²) · V_cell.
    ///
    /// This is the conserved norm — the split-step integrator must
    /// preserve it to machine precision.
    pub fn integrate_norm_squared(&self) -> f64 {
        let n = self.grid.n_cells;
        let mut sum = 0.0;

        for spatial_idx in spatial_indices_iter::<D>(n) {
            let a = self.scalar[IxDyn(&spatial_idx)];
            let b = self.pseudoscalar[IxDyn(&spatial_idx)];
            sum += a * a + b * b;
        }

        sum * self.grid.cell_volume()
    }

    /// Spectral Laplacian, applied componentwise.
    ///
    /// FFT each component, multiply by -|k|², IFFT. Grade-preserving:
    /// even subalgebra in, even subalgebra out.
    pub fn laplacian(&self) -> EvenField<D> {
        let n = self.grid.n_cells;

        let scalar_lap = self.laplacian_component(&self.scalar, n);
        let pseudo_lap = self.laplacian_component(&self.pseudoscalar, n);

        EvenField {
            scalar: scalar_lap,
            pseudoscalar: pseudo_lap,
            metric: self.metric,
            grid: self.grid,
        }
    }

    /// Laplacian of a single spatial array.
    fn laplacian_component(&self, component: &ArrayD<f64>, n: usize) -> ArrayD<f64> {
        let mut hat = fft_forward::<D>(component, n);

        let mut freq = [0usize; D];
        for freq_idx in spatial_indices_iter::<D>(n) {
            freq[..D].copy_from_slice(&freq_idx[..D]);
            let k_sq = self.grid.k_squared(&freq);
            hat[IxDyn(&freq_idx)] *= -k_sq;
        }

        fft_inverse::<D>(&hat, n)
    }

    /// Gradient of each component: [grad(scalar), grad(pseudoscalar)].
    ///
    /// Returns two grade-1 (vector) `Field<D>` values. Each direction
    /// is computed spectrally with Nyquist zeroing for odd-order safety.
    pub fn gradient_components(&self) -> [Field<D>; 2] {
        [
            self.gradient_of_component(&self.scalar),
            self.gradient_of_component(&self.pseudoscalar),
        ]
    }

    /// Spectral gradient of a single spatial array, returned as a grade-1 field.
    fn gradient_of_component(&self, component: &ArrayD<f64>) -> Field<D> {
        let n = self.grid.n_cells;
        let shape = field_shape::<D>(n, 1);
        let mut result_data = ArrayD::<f64>::zeros(IxDyn(&shape));
        let nyquist = n / 2;

        // One forward FFT for this component
        let hat = fft_forward::<D>(component, n);

        // For each spatial direction, multiply by i*k_d (with Nyquist zeroing), IFFT
        for d in 0..D {
            let mut hat_d = hat.clone();

            for freq_idx in spatial_indices_iter::<D>(n) {
                if freq_idx[d] == nyquist {
                    hat_d[IxDyn(&freq_idx)] = Complex::new(0.0, 0.0);
                } else {
                    let k_d = self.grid.wavenumber(freq_idx[d]);
                    hat_d[IxDyn(&freq_idx)] *= Complex::new(0.0, k_d);
                }
            }

            let deriv = fft_inverse::<D>(&hat_d, n);

            // Write into the d-th component of the vector field
            for spatial_idx in spatial_indices_iter::<D>(n) {
                let mut full_idx = spatial_idx.clone();
                full_idx.push(d);
                result_data[IxDyn(&full_idx)] = deriv[IxDyn(&spatial_idx)];
            }
        }

        Field::new(result_data, 1, &self.grid, self.metric)
    }

    /// Kinetic energy density: ½(|∇a|² + |∇b|²).
    ///
    /// Returns the gradient energy density as a scalar field, before
    /// multiplication by the diffusivity parameter.
    pub fn kinetic_energy_density(&self) -> Field<D> {
        let [grad_a, grad_b] = self.gradient_components();

        &(&grad_a.norm_squared() + &grad_b.norm_squared()) * 0.5
    }

    /// Build α from density and velocity potential (Madelung inverse).
    ///
    /// amplitude = sqrt(ρ / m), phase = φ_v / ν
    /// α = (amplitude · cos(phase), amplitude · sin(phase))
    pub fn madelung_inverse(
        density: &Field<D>,
        velocity_potential: &Field<D>,
        mass: f64,
        diffusivity: f64,
    ) -> EvenField<D> {
        assert_eq!(density.grade(), 0, "density must be a scalar field");
        assert_eq!(
            velocity_potential.grade(),
            0,
            "velocity potential must be a scalar field"
        );
        assert_eq!(
            density.grid, velocity_potential.grid,
            "fields must share the same grid"
        );

        let n = density.grid.n_cells;
        let shape: Vec<usize> = vec![n; D];
        let mut scalar = ArrayD::zeros(IxDyn(&shape));
        let mut pseudoscalar = ArrayD::zeros(IxDyn(&shape));

        for spatial_idx in spatial_indices_iter::<D>(n) {
            let rho = density.data[IxDyn(&spatial_idx)];
            let phi_v = velocity_potential.data[IxDyn(&spatial_idx)];
            let amplitude = (rho / mass).sqrt();
            let phase = phi_v / diffusivity;
            scalar[IxDyn(&spatial_idx)] = amplitude * phase.cos();
            pseudoscalar[IxDyn(&spatial_idx)] = amplitude * phase.sin();
        }

        EvenField {
            scalar,
            pseudoscalar,
            metric: density.metric,
            grid: density.grid,
        }
    }

    /// Extract the Madelung velocity as a grade-1 vector field.
    ///
    /// v_d(x) = ν / (a² + b²) · (a ∂_d b - b ∂_d a)
    ///
    /// Where |α|² < 1e-30, the denominator is clamped to avoid
    /// division by zero in vacuum regions.
    pub fn madelung_velocity(&self, diffusivity: f64) -> Field<D> {
        let [grad_a, grad_b] = self.gradient_components();
        let n = self.grid.n_cells;
        let shape = field_shape::<D>(n, 1);
        let mut result_data = ArrayD::<f64>::zeros(IxDyn(&shape));

        for spatial_idx in spatial_indices_iter::<D>(n) {
            let a = self.scalar[IxDyn(&spatial_idx)];
            let b = self.pseudoscalar[IxDyn(&spatial_idx)];
            let norm_sq = (a * a + b * b).max(1e-30);

            for d in 0..D {
                let mut full_idx = spatial_idx.clone();
                full_idx.push(d);
                let da_d = grad_a.data[IxDyn(&full_idx)];
                let db_d = grad_b.data[IxDyn(&full_idx)];
                result_data[IxDyn(&full_idx)] = diffusivity * (a * db_d - b * da_d) / norm_sq;
            }
        }

        Field::new(result_data, 1, &self.grid, self.metric)
    }
}

impl<const D: usize> Clone for EvenField<D> {
    fn clone(&self) -> Self {
        Self {
            scalar: self.scalar.clone(),
            pseudoscalar: self.pseudoscalar.clone(),
            metric: self.metric,
            grid: self.grid,
        }
    }
}

// =============================================================================
// Arithmetic
// =============================================================================

impl<const D: usize> std::ops::Add for &EvenField<D> {
    type Output = EvenField<D>;

    fn add(self, rhs: &EvenField<D>) -> EvenField<D> {
        EvenField {
            scalar: &self.scalar + &rhs.scalar,
            pseudoscalar: &self.pseudoscalar + &rhs.pseudoscalar,
            metric: self.metric,
            grid: self.grid,
        }
    }
}

impl<const D: usize> std::ops::Sub for &EvenField<D> {
    type Output = EvenField<D>;

    fn sub(self, rhs: &EvenField<D>) -> EvenField<D> {
        EvenField {
            scalar: &self.scalar - &rhs.scalar,
            pseudoscalar: &self.pseudoscalar - &rhs.pseudoscalar,
            metric: self.metric,
            grid: self.grid,
        }
    }
}

impl<const D: usize> std::ops::Mul<f64> for &EvenField<D> {
    type Output = EvenField<D>;

    fn mul(self, rhs: f64) -> EvenField<D> {
        EvenField {
            scalar: &self.scalar * rhs,
            pseudoscalar: &self.pseudoscalar * rhs,
            metric: self.metric,
            grid: self.grid,
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Set the pseudoscalar component in a grade-D tensor.
///
/// The pseudoscalar e1∧e2∧...∧eD is stored as the fully antisymmetric
/// tensor with value ε_{012...D-1} * coeff.
fn set_pseudoscalar_component<const D: usize>(data: &mut ArrayD<f64>, coeff: f64) {
    // Generate all permutations of [0, 1, ..., D-1] and set with appropriate sign
    let identity: Vec<usize> = (0..D).collect();
    set_antisymmetric_from_canonical::<D>(data, &identity, coeff);
}

/// Set all antisymmetric permutations of a canonical index.
fn set_antisymmetric_from_canonical<const D: usize>(
    data: &mut ArrayD<f64>,
    canonical: &[usize],
    coeff: f64,
) {
    let perms = permutations(canonical);
    for perm in &perms {
        let sign = permutation_sign(perm, canonical);
        data[IxDyn(perm)] = sign as f64 * coeff;
    }
}

/// Generate all permutations of a slice.
fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut result = Vec::new();
    for (k, &item) in items.iter().enumerate() {
        let rest: Vec<usize> = items
            .iter()
            .enumerate()
            .filter(|&(m, _)| m != k)
            .map(|(_, &v)| v)
            .collect();
        for mut perm in permutations(&rest) {
            perm.insert(0, item);
            result.push(perm);
        }
    }
    result
}

/// Compute the sign of a permutation relative to a canonical ordering.
fn permutation_sign(perm: &[usize], canonical: &[usize]) -> i32 {
    // Count inversions
    let mut inversions = 0;
    for a in 0..perm.len() {
        for b in (a + 1)..perm.len() {
            let pos_a = canonical.iter().position(|&x| x == perm[a]).unwrap();
            let pos_b = canonical.iter().position(|&x| x == perm[b]).unwrap();
            if pos_a > pos_b {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 { 1 } else { -1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::euclidean;

    #[test]
    fn zeros_is_zero() {
        let g = euclidean::<3>();
        let grid = Grid::<3>::new(4, 1.0);
        let f = EvenField::zeros(&grid, g);
        assert!(f.scalar.iter().all(|&x| x == 0.0));
        assert!(f.pseudoscalar.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn from_fn_stores_values() {
        let g = euclidean::<3>();
        let grid = Grid::<3>::new(4, 1.0);
        let f = EvenField::from_fn(&grid, g, |_| (1.0, 2.0));

        assert!((f.scalar[IxDyn(&[0, 0, 0])] - 1.0).abs() < 1e-15);
        assert!((f.pseudoscalar[IxDyn(&[0, 0, 0])] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn rev_flips_pseudoscalar() {
        let g = euclidean::<3>();
        let grid = Grid::<3>::new(4, 1.0);
        let f = EvenField::from_fn(&grid, g, |_| (3.0, 5.0));
        let r = f.rev();

        assert!((r.scalar[IxDyn(&[0, 0, 0])] - 3.0).abs() < 1e-15);
        assert!((r.pseudoscalar[IxDyn(&[0, 0, 0])] + 5.0).abs() < 1e-15);
    }

    #[test]
    fn mul_complex_arithmetic() {
        let g = euclidean::<3>();
        let grid = Grid::<3>::new(2, 1.0);
        // (2 + 3I)(4 + 5I) = (8-15) + (10+12)I = -7 + 22I
        let f1 = EvenField::from_fn(&grid, g, |_| (2.0, 3.0));
        let f2 = EvenField::from_fn(&grid, g, |_| (4.0, 5.0));
        let product = f1.mul(&f2);

        assert!((product.scalar[IxDyn(&[0, 0, 0])] + 7.0).abs() < 1e-12);
        assert!((product.pseudoscalar[IxDyn(&[0, 0, 0])] - 22.0).abs() < 1e-12);
    }

    #[test]
    fn permutation_sign_identity() {
        assert_eq!(permutation_sign(&[0, 1, 2], &[0, 1, 2]), 1);
    }

    #[test]
    fn permutation_sign_transposition() {
        assert_eq!(permutation_sign(&[1, 0, 2], &[0, 1, 2]), -1);
    }

    #[test]
    fn permutation_sign_cyclic() {
        // (0,1,2) -> (2,0,1) is two transpositions: even
        assert_eq!(permutation_sign(&[2, 0, 1], &[0, 1, 2]), 1);
    }
}
