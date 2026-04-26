use nalgebra::DMatrix;
use ndarray::{ArrayD, IxDyn};

use crate::metric::Metric;
use crate::util::{flatten_index, indices_iter};
use crate::vector::Vector;

/// A linear operator mapping grade-j vectors to grade-k vectors.
///
/// Stores a rank-(j+k) tensor with the first k indices as output and the
/// last j indices as input. Application contracts the input indices with
/// the operand's tensor.
///
/// For decompositions (SVD, pseudoinverse, solve), the tensor is flattened
/// to a D^k × D^j matrix, operated on via nalgebra, and reshaped back.
#[derive(Debug, Clone)]
pub struct Operator<const D: usize> {
    /// Tensor data of shape [D; out_grade + in_grade].
    data: ArrayD<f64>,
    /// Grade of the input vectors.
    in_grade: usize,
    /// Grade of the output vectors.
    out_grade: usize,
    /// Metric defining the inner product structure.
    pub metric: Metric<D>,
}

impl<const D: usize> Operator<D> {
    /// Create an operator from a rank-(k+j) tensor.
    ///
    /// The first `out_grade` indices are output, the last `in_grade` are input.
    pub fn new(data: ArrayD<f64>, in_grade: usize, out_grade: usize, metric: Metric<D>) -> Self {
        let expected_shape: Vec<usize> = vec![D; out_grade + in_grade];
        assert_eq!(
            data.shape(),
            expected_shape.as_slice(),
            "operator data shape {:?} does not match expected {:?}",
            data.shape(),
            expected_shape,
        );

        Self {
            data,
            in_grade,
            out_grade,
            metric,
        }
    }

    /// Create an operator from a flat matrix (D^out_grade rows, D^in_grade columns).
    ///
    /// The matrix is reshaped to a rank-(k+j) tensor using row-major multi-index layout.
    pub fn from_matrix(
        matrix: &DMatrix<f64>,
        in_grade: usize,
        out_grade: usize,
        metric: Metric<D>,
    ) -> Self {
        let out_flat = D.pow(out_grade as u32);
        let in_flat = D.pow(in_grade as u32);
        assert_eq!(
            (matrix.nrows(), matrix.ncols()),
            (out_flat, in_flat),
            "matrix shape ({}, {}) does not match expected ({}, {})",
            matrix.nrows(),
            matrix.ncols(),
            out_flat,
            in_flat,
        );

        let rank = out_grade + in_grade;
        let shape: Vec<usize> = vec![D; rank];
        let mut data = ArrayD::zeros(IxDyn(&shape));

        for out_idx in indices_iter(D, out_grade) {
            let row = flatten_index(&out_idx, D);
            for in_idx in indices_iter(D, in_grade) {
                let col = flatten_index(&in_idx, D);
                let mut full_idx = out_idx.clone();
                full_idx.extend_from_slice(&in_idx);
                data[IxDyn(&full_idx)] = matrix[(row, col)];
            }
        }

        Self {
            data,
            in_grade,
            out_grade,
            metric,
        }
    }

    /// Flatten the operator tensor to a D^k × D^j matrix.
    pub fn to_matrix(&self) -> DMatrix<f64> {
        let out_flat = D.pow(self.out_grade as u32);
        let in_flat = D.pow(self.in_grade as u32);
        let mut matrix = DMatrix::zeros(out_flat, in_flat);

        for out_idx in indices_iter(D, self.out_grade) {
            let row = flatten_index(&out_idx, D);
            for in_idx in indices_iter(D, self.in_grade) {
                let col = flatten_index(&in_idx, D);
                let mut full_idx = out_idx.clone();
                full_idx.extend_from_slice(&in_idx);
                matrix[(row, col)] = self.data[IxDyn(&full_idx)];
            }
        }

        matrix
    }

    /// Grade of the input vectors.
    pub fn in_grade(&self) -> usize {
        self.in_grade
    }

    /// Grade of the output vectors.
    pub fn out_grade(&self) -> usize {
        self.out_grade
    }

    /// Dimension of the underlying vector space.
    pub fn dim(&self) -> usize {
        D
    }

    /// Apply the operator to a k-vector.
    ///
    /// Contracts the last in_grade indices of the operator tensor with
    /// all indices of the input vector.
    pub fn apply(&self, v: &Vector<D>) -> Vector<D> {
        assert_eq!(
            v.grade(),
            self.in_grade,
            "input grade {} does not match operator input grade {}",
            v.grade(),
            self.in_grade,
        );

        let out_shape: Vec<usize> = vec![D; self.out_grade];
        let mut result = if self.out_grade == 0 {
            ArrayD::from_elem(IxDyn(&[]), 0.0)
        } else {
            ArrayD::zeros(IxDyn(&out_shape))
        };

        for out_idx in indices_iter(D, self.out_grade) {
            let mut sum = 0.0;

            for in_idx in indices_iter(D, self.in_grade) {
                let v_val = v.data[IxDyn(&in_idx)];
                if v_val.abs() < 1e-15 {
                    continue;
                }

                let mut full_idx = out_idx.clone();
                full_idx.extend_from_slice(&in_idx);
                sum += self.data[IxDyn(&full_idx)] * v_val;
            }

            result[IxDyn(&out_idx)] = sum;
        }

        Vector::new(result, self.out_grade, self.metric)
    }

    /// Compose two operators: (self ∘ other)(v) = self(other(v)).
    ///
    /// Requires self.in_grade == other.out_grade.
    pub fn compose(&self, other: &Operator<D>) -> Operator<D> {
        assert_eq!(
            self.in_grade, other.out_grade,
            "cannot compose: left input grade {} != right output grade {}",
            self.in_grade, other.out_grade,
        );

        let left = self.to_matrix();
        let right = other.to_matrix();
        let composed = left * right;

        Operator::from_matrix(&composed, other.in_grade, self.out_grade, self.metric)
    }

    /// Adjoint: swap input and output index groups.
    ///
    /// The adjoint L† satisfies <L(u), v> = <u, L†(v)> with respect to
    /// the standard inner product on antisymmetric tensors.
    pub fn adjoint(&self) -> Operator<D> {
        let matrix = self.to_matrix();
        let adj = matrix.transpose();

        Operator::from_matrix(&adj, self.out_grade, self.in_grade, self.metric)
    }

    /// Singular value decomposition: L = U diag(σ) V†.
    ///
    /// Returns (U, singular_values, Vt) where:
    /// - U maps from a reduced space to the output grade
    /// - Vt maps from the input grade to the reduced space
    /// - singular_values are sorted descending
    pub fn svd(&self) -> (DMatrix<f64>, Vec<f64>, DMatrix<f64>) {
        let matrix = self.to_matrix();
        let svd = matrix.svd(true, true);

        let u = svd.u.expect("SVD should produce U");
        let sigma: Vec<f64> = svd.singular_values.iter().copied().collect();
        let vt = svd.v_t.expect("SVD should produce V^T");

        (u, sigma, vt)
    }

    /// Moore-Penrose pseudoinverse: L⁺.
    ///
    /// Satisfies L L⁺ L = L and L⁺ L L⁺ = L⁺.
    pub fn pseudoinverse(&self) -> Operator<D> {
        let (u, sigma, vt) = self.svd();

        let tol = 1e-12 * sigma.first().copied().unwrap_or(0.0);
        let mut sigma_inv = DMatrix::zeros(vt.nrows(), u.ncols());
        for (m, &s) in sigma.iter().enumerate() {
            if s > tol {
                sigma_inv[(m, m)] = 1.0 / s;
            }
        }

        let pinv = vt.transpose() * sigma_inv * u.transpose();

        Operator::from_matrix(&pinv, self.out_grade, self.in_grade, self.metric)
    }

    /// Solve the inverse problem: find x such that L(x) ≈ y.
    ///
    /// Uses least-squares via SVD. For overdetermined systems, finds the
    /// minimum-residual solution. For underdetermined, finds the minimum-norm.
    pub fn solve(&self, y: &Vector<D>) -> Vector<D> {
        assert_eq!(
            y.grade(),
            self.out_grade,
            "target grade {} does not match operator output grade {}",
            y.grade(),
            self.out_grade,
        );

        let pinv = self.pseudoinverse();

        pinv.apply(y)
    }
}

// =============================================================================
// Operator Overloads
// =============================================================================

// Operator * Vector -> Vector (application)
impl<const D: usize> std::ops::Mul<&Vector<D>> for &Operator<D> {
    type Output = Vector<D>;
    fn mul(self, rhs: &Vector<D>) -> Vector<D> {
        self.apply(rhs)
    }
}

impl<const D: usize> std::ops::Mul<Vector<D>> for Operator<D> {
    type Output = Vector<D>;
    fn mul(self, rhs: Vector<D>) -> Vector<D> {
        self.apply(&rhs)
    }
}

impl<const D: usize> std::ops::Mul<Vector<D>> for &Operator<D> {
    type Output = Vector<D>;
    fn mul(self, rhs: Vector<D>) -> Vector<D> {
        self.apply(&rhs)
    }
}

impl<const D: usize> std::ops::Mul<&Vector<D>> for Operator<D> {
    type Output = Vector<D>;
    fn mul(self, rhs: &Vector<D>) -> Vector<D> {
        self.apply(rhs)
    }
}

// Operator * Operator -> Operator (composition)
impl<const D: usize> std::ops::Mul for &Operator<D> {
    type Output = Operator<D>;
    fn mul(self, rhs: &Operator<D>) -> Operator<D> {
        self.compose(rhs)
    }
}

impl<const D: usize> std::ops::Mul for Operator<D> {
    type Output = Operator<D>;
    fn mul(self, rhs: Operator<D>) -> Operator<D> {
        self.compose(&rhs)
    }
}

impl<const D: usize> std::ops::Mul<Operator<D>> for &Operator<D> {
    type Output = Operator<D>;
    fn mul(self, rhs: Operator<D>) -> Operator<D> {
        self.compose(&rhs)
    }
}

impl<const D: usize> std::ops::Mul<&Operator<D>> for Operator<D> {
    type Output = Operator<D>;
    fn mul(self, rhs: &Operator<D>) -> Operator<D> {
        self.compose(rhs)
    }
}
