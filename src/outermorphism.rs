use nalgebra::DMatrix;
use ndarray::{ArrayD, IxDyn};

use crate::metric::Metric;
use crate::multivector::MultiVector;
use crate::util::indices_iter;
use crate::vector::Vector;

/// An outermorphism: a grade-1 linear map extended to all grades.
///
/// Stores a d×d matrix A that acts on grade-1 vectors via matrix-vector
/// multiplication. For grade-k elements, the action is the k-th exterior
/// power: k copies of A contract with the k tensor indices.
///
/// The defining property is that outermorphisms preserve the wedge product:
/// A(u ∧ v) = A(u) ∧ A(v).
#[derive(Debug, Clone)]
pub struct Outermorphism<const D: usize> {
    /// The d×d matrix representing the grade-1 linear map.
    matrix: DMatrix<f64>,
    /// Metric defining the inner product structure.
    pub metric: Metric<D>,
}

impl<const D: usize> Outermorphism<D> {
    /// Create an outermorphism from a d×d matrix.
    pub fn new(matrix: DMatrix<f64>, metric: Metric<D>) -> Self {
        assert_eq!(
            (matrix.nrows(), matrix.ncols()),
            (D, D),
            "outermorphism matrix must be {}×{}, got {}×{}",
            D,
            D,
            matrix.nrows(),
            matrix.ncols(),
        );

        Self { matrix, metric }
    }

    /// Create an outermorphism from a row-major slice of d² elements.
    pub fn from_row_slice(data: &[f64], metric: Metric<D>) -> Self {
        assert_eq!(
            data.len(),
            D * D,
            "expected {} elements for {}×{} matrix, got {}",
            D * D,
            D,
            D,
            data.len(),
        );
        let matrix = DMatrix::from_row_slice(D, D, data);

        Self { matrix, metric }
    }

    /// The identity outermorphism.
    pub fn identity(metric: Metric<D>) -> Self {
        Self {
            matrix: DMatrix::identity(D, D),
            metric,
        }
    }

    /// Build an outermorphism from the images of the basis vectors.
    ///
    /// Column m of the matrix is the image of the m-th basis vector.
    pub fn from_columns(columns: &[Vector<D>], metric: Metric<D>) -> Self {
        assert_eq!(
            columns.len(),
            D,
            "expected {} column vectors, got {}",
            D,
            columns.len(),
        );
        for (m, col) in columns.iter().enumerate() {
            assert_eq!(
                col.grade(),
                1,
                "column {} must be grade 1, got grade {}",
                m,
                col.grade(),
            );
        }

        let mut matrix = DMatrix::zeros(D, D);
        for m in 0..D {
            for n in 0..D {
                matrix[(n, m)] = columns[m].component(&[n]);
            }
        }

        Self { matrix, metric }
    }

    /// Access the underlying d×d matrix.
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Apply the outermorphism to a k-vector.
    ///
    /// For grade 0 (scalar): returns unchanged.
    /// For grade 1 (vector): matrix-vector product.
    /// For grade k: the k-th exterior power, contracting k copies of the
    /// matrix with the k tensor indices of the blade.
    pub fn apply(&self, v: &Vector<D>) -> Vector<D> {
        let k = v.grade();

        if k == 0 {
            return v.clone();
        }

        if k == 1 {
            return self.apply_grade_1(v);
        }

        self.apply_grade_k(v, k)
    }

    /// Apply to a grade-1 vector: matrix-vector product.
    fn apply_grade_1(&self, v: &Vector<D>) -> Vector<D> {
        let mut result = ArrayD::zeros(IxDyn(&[D]));

        for m in 0..D {
            let mut sum = 0.0;
            for n in 0..D {
                sum += self.matrix[(m, n)] * v.component(&[n]);
            }
            result[IxDyn(&[m])] = sum;
        }

        Vector::new(result, 1, self.metric)
    }

    /// Apply the k-th exterior power to a grade-k blade.
    ///
    /// (∧^k A)(B)^{i1...ik} = A^{i1}_{m1} ... A^{ik}_{mk} B^{m1...mk}
    ///
    /// Implementation: contract each tensor axis successively with the matrix.
    fn apply_grade_k(&self, v: &Vector<D>, k: usize) -> Vector<D> {
        let mut result = ArrayD::zeros(IxDyn(&vec![D; k]));

        for out_idx in indices_iter(D, k) {
            let mut sum = 0.0;

            for in_idx in indices_iter(D, k) {
                let v_val = v.data[IxDyn(&in_idx)];
                if v_val.abs() < 1e-15 {
                    continue;
                }

                // Product of matrix entries: A^{i1}_{m1} * A^{i2}_{m2} * ... * A^{ik}_{mk}
                let matrix_factor: f64 = (0..k)
                    .map(|p| self.matrix[(out_idx[p], in_idx[p])])
                    .product();

                sum += matrix_factor * v_val;
            }

            result[IxDyn(&out_idx)] = sum;
        }

        Vector::new(result, k, self.metric)
    }

    /// Apply the outermorphism to a multivector, grade by grade.
    pub fn apply_mv(&self, m: &MultiVector<D>) -> MultiVector<D> {
        let mut components = std::collections::HashMap::new();

        for (&k, component) in m.components() {
            let transformed = self.apply(component);
            if !transformed.is_zero(1e-15) {
                components.insert(k, transformed);
            }
        }

        MultiVector::from_components(components, self.metric)
    }

    /// Compose two outermorphisms: (self ∘ other)(v) = self(other(v)).
    ///
    /// The composed matrix is self.matrix * other.matrix.
    pub fn compose(&self, other: &Outermorphism<D>) -> Outermorphism<D> {
        Outermorphism {
            matrix: &self.matrix * &other.matrix,
            metric: self.metric,
        }
    }

    /// Transpose of the outermorphism.
    pub fn transpose(&self) -> Outermorphism<D> {
        Outermorphism {
            matrix: self.matrix.transpose(),
            metric: self.metric,
        }
    }

    /// Determinant of the underlying matrix.
    pub fn det(&self) -> f64 {
        self.matrix.determinant()
    }

    /// Inverse outermorphism, if the matrix is invertible.
    pub fn inv(&self) -> Option<Outermorphism<D>> {
        self.matrix.clone().try_inverse().map(|inv| Outermorphism {
            matrix: inv,
            metric: self.metric,
        })
    }

    /// Dimension of the underlying vector space.
    pub fn dim(&self) -> usize {
        D
    }
}

// =============================================================================
// Operator Overloads
// =============================================================================

// Outermorphism * Vector -> Vector (apply)
impl<const D: usize> std::ops::Mul<&Vector<D>> for &Outermorphism<D> {
    type Output = Vector<D>;
    fn mul(self, rhs: &Vector<D>) -> Vector<D> {
        self.apply(rhs)
    }
}

impl<const D: usize> std::ops::Mul<Vector<D>> for Outermorphism<D> {
    type Output = Vector<D>;
    fn mul(self, rhs: Vector<D>) -> Vector<D> {
        self.apply(&rhs)
    }
}

impl<const D: usize> std::ops::Mul<Vector<D>> for &Outermorphism<D> {
    type Output = Vector<D>;
    fn mul(self, rhs: Vector<D>) -> Vector<D> {
        self.apply(&rhs)
    }
}

impl<const D: usize> std::ops::Mul<&Vector<D>> for Outermorphism<D> {
    type Output = Vector<D>;
    fn mul(self, rhs: &Vector<D>) -> Vector<D> {
        self.apply(rhs)
    }
}

// Outermorphism * MultiVector -> MultiVector (apply grade-by-grade)
impl<const D: usize> std::ops::Mul<&MultiVector<D>> for &Outermorphism<D> {
    type Output = MultiVector<D>;
    fn mul(self, rhs: &MultiVector<D>) -> MultiVector<D> {
        self.apply_mv(rhs)
    }
}

impl<const D: usize> std::ops::Mul<MultiVector<D>> for Outermorphism<D> {
    type Output = MultiVector<D>;
    fn mul(self, rhs: MultiVector<D>) -> MultiVector<D> {
        self.apply_mv(&rhs)
    }
}

impl<const D: usize> std::ops::Mul<MultiVector<D>> for &Outermorphism<D> {
    type Output = MultiVector<D>;
    fn mul(self, rhs: MultiVector<D>) -> MultiVector<D> {
        self.apply_mv(&rhs)
    }
}

impl<const D: usize> std::ops::Mul<&MultiVector<D>> for Outermorphism<D> {
    type Output = MultiVector<D>;
    fn mul(self, rhs: &MultiVector<D>) -> MultiVector<D> {
        self.apply_mv(rhs)
    }
}

// Outermorphism * Outermorphism -> Outermorphism (composition)
impl<const D: usize> std::ops::Mul for &Outermorphism<D> {
    type Output = Outermorphism<D>;
    fn mul(self, rhs: &Outermorphism<D>) -> Outermorphism<D> {
        self.compose(rhs)
    }
}

impl<const D: usize> std::ops::Mul for Outermorphism<D> {
    type Output = Outermorphism<D>;
    fn mul(self, rhs: Outermorphism<D>) -> Outermorphism<D> {
        self.compose(&rhs)
    }
}

impl<const D: usize> std::ops::Mul<Outermorphism<D>> for &Outermorphism<D> {
    type Output = Outermorphism<D>;
    fn mul(self, rhs: Outermorphism<D>) -> Outermorphism<D> {
        self.compose(&rhs)
    }
}

impl<const D: usize> std::ops::Mul<&Outermorphism<D>> for Outermorphism<D> {
    type Output = Outermorphism<D>;
    fn mul(self, rhs: &Outermorphism<D>) -> Outermorphism<D> {
        self.compose(rhs)
    }
}

// Scalar multiplication
impl<const D: usize> std::ops::Mul<f64> for &Outermorphism<D> {
    type Output = Outermorphism<D>;
    fn mul(self, rhs: f64) -> Outermorphism<D> {
        Outermorphism {
            matrix: &self.matrix * rhs,
            metric: self.metric,
        }
    }
}

impl<const D: usize> std::ops::Mul<f64> for Outermorphism<D> {
    type Output = Outermorphism<D>;
    fn mul(self, rhs: f64) -> Outermorphism<D> {
        &self * rhs
    }
}

impl<const D: usize> std::ops::Mul<&Outermorphism<D>> for f64 {
    type Output = Outermorphism<D>;
    fn mul(self, rhs: &Outermorphism<D>) -> Outermorphism<D> {
        rhs * self
    }
}

impl<const D: usize> std::ops::Mul<Outermorphism<D>> for f64 {
    type Output = Outermorphism<D>;
    fn mul(self, rhs: Outermorphism<D>) -> Outermorphism<D> {
        &rhs * self
    }
}
