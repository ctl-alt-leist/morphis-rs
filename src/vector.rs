use ndarray::{ArrayD, IxDyn};

use crate::metric::Metric;

/// A k-vector in geometric algebra.
///
/// Represents a homogeneous multivector of pure grade k in a D-dimensional
/// vector space. Storage is a full antisymmetric tensor of shape [D; K],
/// where antisymmetry is an invariant maintained by constructors and
/// operations.
///
/// Scalars have grade 0, vectors grade 1, bivectors grade 2, etc.
#[derive(Debug, Clone)]
pub struct Vector<const D: usize> {
    /// Antisymmetric tensor components. Shape is [D] repeated `grade` times.
    pub data: ArrayD<f64>,
    /// Grade of this k-vector (0 = scalar, 1 = vector, 2 = bivector, ...).
    grade: usize,
    /// Metric defining the inner product structure.
    pub metric: Metric<D>,
}

impl<const D: usize> Vector<D> {
    /// Create a new k-vector from raw antisymmetric tensor data.
    ///
    /// The data shape must be [D] repeated `grade` times. Antisymmetry is
    /// assumed — the caller is responsible for providing valid data.
    pub fn new(data: ArrayD<f64>, grade: usize, metric: Metric<D>) -> Self {
        let expected_shape: Vec<usize> = vec![D; grade];
        assert_eq!(
            data.shape(),
            expected_shape.as_slice(),
            "data shape {:?} does not match expected {:?} for grade-{} vector in {}-dimensional space",
            data.shape(),
            expected_shape,
            grade,
            D,
        );

        Self {
            data,
            grade,
            metric,
        }
    }

    /// Create a zero k-vector of the given grade.
    pub fn zero(grade: usize, metric: Metric<D>) -> Self {
        let shape: Vec<usize> = vec![D; grade];
        let data = ArrayD::zeros(IxDyn(&shape));

        Self {
            data,
            grade,
            metric,
        }
    }

    /// Create a scalar (grade-0) k-vector.
    pub fn scalar(value: f64, metric: Metric<D>) -> Self {
        let data = ArrayD::from_elem(IxDyn(&[]), value);

        Self {
            data,
            grade: 0,
            metric,
        }
    }

    /// Grade of this k-vector.
    pub fn grade(&self) -> usize {
        self.grade
    }

    /// Dimension of the underlying vector space.
    pub fn dim(&self) -> usize {
        D
    }

    /// Whether this k-vector is zero (all components vanish).
    pub fn is_zero(&self, tol: f64) -> bool {
        self.data.iter().all(|x| x.abs() < tol)
    }

    /// Reverse: reverses the order of grade-1 factors.
    ///
    /// For a grade-k vector: rev(v) = (-1)^{k(k-1)/2} v
    pub fn rev(&self) -> Self {
        let k = self.grade;
        // k * (k-1) / 2 gives the number of transpositions.
        // For k = 0 or 1, sign is always +1.
        let sign = if k < 2 || (k * (k - 1) / 2).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };

        Self {
            data: &self.data * sign,
            grade: self.grade,
            metric: self.metric,
        }
    }

    /// Squared norm: <v ~v>_0 with appropriate normalization.
    ///
    /// For a grade-k vector, computes g_{a1 b1} ... g_{ak bk} v^{a1...ak} v^{b1...bk} / k!
    pub fn norm_squared(&self) -> f64 {
        let k = self.grade;

        if k == 0 {
            let s = self.data[IxDyn(&[])];

            return s * s;
        }

        let mut sum = 0.0;
        for idx in indices_iter(D, k) {
            let val = self.data[IxDyn(&idx)];
            if val.abs() < 1e-15 {
                continue;
            }

            // Compute metric factor: product of g_{a_m a_m} for each index
            let metric_factor: f64 = idx.iter().map(|&a| self.metric.diag[a]).product();
            sum += val * val * metric_factor;
        }

        sum / factorial(k) as f64
    }

    /// Norm: sqrt(|norm_squared|)
    pub fn norm(&self) -> f64 {
        self.norm_squared().abs().sqrt()
    }

    /// Normalized copy: v / ||v||
    ///
    /// Returns None if the vector has zero norm.
    pub fn normalize(&self) -> Option<Self> {
        let n = self.norm();
        if n < 1e-15 {
            return None;
        }

        Some(Self {
            data: &self.data / n,
            grade: self.grade,
            metric: self.metric,
        })
    }

    /// Component access for a single multi-index.
    pub fn component(&self, indices: &[usize]) -> f64 {
        self.data[IxDyn(indices)]
    }
}

// =============================================================================
// Arithmetic Operators
// =============================================================================

impl<const D: usize> std::ops::Add for &Vector<D> {
    type Output = Vector<D>;

    fn add(self, rhs: &Vector<D>) -> Vector<D> {
        assert_eq!(
            self.grade, rhs.grade,
            "cannot add grade-{} and grade-{} vectors",
            self.grade, rhs.grade,
        );

        Vector {
            data: &self.data + &rhs.data,
            grade: self.grade,
            metric: self.metric,
        }
    }
}

impl<const D: usize> std::ops::Sub for &Vector<D> {
    type Output = Vector<D>;

    fn sub(self, rhs: &Vector<D>) -> Vector<D> {
        assert_eq!(
            self.grade, rhs.grade,
            "cannot subtract grade-{} and grade-{} vectors",
            self.grade, rhs.grade,
        );

        Vector {
            data: &self.data - &rhs.data,
            grade: self.grade,
            metric: self.metric,
        }
    }
}

impl<const D: usize> std::ops::Neg for &Vector<D> {
    type Output = Vector<D>;

    fn neg(self) -> Vector<D> {
        Vector {
            data: -&self.data,
            grade: self.grade,
            metric: self.metric,
        }
    }
}

/// Scalar multiplication: scalar * vector
impl<const D: usize> std::ops::Mul<f64> for &Vector<D> {
    type Output = Vector<D>;

    fn mul(self, rhs: f64) -> Vector<D> {
        Vector {
            data: &self.data * rhs,
            grade: self.grade,
            metric: self.metric,
        }
    }
}

/// Scalar division: vector / scalar
impl<const D: usize> std::ops::Div<f64> for &Vector<D> {
    type Output = Vector<D>;

    fn div(self, rhs: f64) -> Vector<D> {
        Vector {
            data: &self.data / rhs,
            grade: self.grade,
            metric: self.metric,
        }
    }
}

impl<const D: usize> PartialEq for Vector<D> {
    fn eq(&self, other: &Self) -> bool {
        self.grade == other.grade && self.data == other.data
    }
}

// =============================================================================
// Basis Constructors
// =============================================================================

/// Construct the standard basis vectors e_0, e_1, ..., e_{D-1}.
pub fn basis<const D: usize>(metric: Metric<D>) -> [Vector<D>; D] {
    std::array::from_fn(|m| {
        let mut data = ArrayD::zeros(IxDyn(&[D]));
        data[IxDyn(&[m])] = 1.0;

        Vector::new(data, 1, metric)
    })
}

// =============================================================================
// Helpers
// =============================================================================

/// Iterate over all multi-indices of length `k` with each index in [0, d).
fn indices_iter(d: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }

    let mut result = Vec::new();
    let mut current = vec![0usize; k];
    loop {
        result.push(current.clone());

        // Increment the multi-index (odometer style)
        let mut pos = k - 1;
        loop {
            current[pos] += 1;
            if current[pos] < d {
                break;
            }
            current[pos] = 0;
            if pos == 0 {
                return result;
            }
            pos -= 1;
        }
    }
}

/// Factorial of n.
pub(crate) fn factorial(n: usize) -> usize {
    (1..=n).product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::euclidean;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn basis_vectors_3d() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        assert_eq!(e[0].component(&[0]), 1.0);
        assert_eq!(e[0].component(&[1]), 0.0);
        assert_eq!(e[0].component(&[2]), 0.0);

        assert_eq!(e[1].component(&[0]), 0.0);
        assert_eq!(e[1].component(&[1]), 1.0);

        assert_eq!(e[2].component(&[2]), 1.0);
    }

    #[test]
    fn zero_vector() {
        let g: Metric<3> = euclidean();
        let v = Vector::<3>::zero(1, g);

        assert!(v.is_zero(1e-15));
        assert_eq!(v.grade(), 1);
        assert_eq!(v.dim(), 3);
    }

    #[test]
    fn scalar_construction() {
        let g: Metric<3> = euclidean();
        let s = Vector::<3>::scalar(5.0, g);

        assert_eq!(s.grade(), 0);
        assert_eq!(s.data[IxDyn(&[])], 5.0);
    }

    #[test]
    fn vector_addition() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let v = &e[0] + &e[1];

        assert_eq!(v.component(&[0]), 1.0);
        assert_eq!(v.component(&[1]), 1.0);
        assert_eq!(v.component(&[2]), 0.0);
    }

    #[test]
    fn vector_subtraction() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let v = &e[1] - &e[0];

        assert_eq!(v.component(&[0]), -1.0);
        assert_eq!(v.component(&[1]), 1.0);
    }

    #[test]
    fn vector_negation() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let v = -&e[0];

        assert_eq!(v.component(&[0]), -1.0);
    }

    #[test]
    fn scalar_multiplication() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let v = &e[0] * 3.0;

        assert_eq!(v.component(&[0]), 3.0);
    }

    #[test]
    fn euclidean_norm_grade1() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let v = &(&e[0] * 3.0) + &(&e[1] * 4.0);

        assert!((v.norm() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn normalize_unit_vector() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let v = &(&e[0] * 3.0) + &(&e[1] * 4.0);
        let u = v.normalize().unwrap();

        assert!((u.norm() - 1.0).abs() < 1e-12);
        assert!((u.component(&[0]) - 0.6).abs() < 1e-12);
        assert!((u.component(&[1]) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn reverse_involution() {
        let g: Metric<3> = euclidean();
        let mut data = ArrayD::zeros(IxDyn(&[3, 3]));
        data[[0, 1]] = 1.0;
        data[[1, 0]] = -1.0;
        let b = Vector::<3>::new(data, 2, g);

        // Grade-2: rev sign = (-1)^{2*1/2} = -1
        let b_rev = b.rev();
        assert_eq!(b_rev.component(&[0, 1]), -1.0);
        assert_eq!(b_rev.component(&[1, 0]), 1.0);

        // Double reverse restores original
        let b_rev_rev = b_rev.rev();
        assert_eq!(b_rev_rev, b);
    }

    #[test]
    fn reverse_grade1_is_identity() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        // Grade-1: rev sign = (-1)^0 = +1
        let e0_rev = e[0].rev();
        assert_eq!(e0_rev, e[0]);
    }

    #[test]
    fn reverse_grade3_sign() {
        let g: Metric<3> = euclidean();
        let mut data = ArrayD::zeros(IxDyn(&[3, 3, 3]));
        data[[0, 1, 2]] = 1.0;
        data[[1, 0, 2]] = -1.0;
        data[[0, 2, 1]] = -1.0;
        data[[2, 1, 0]] = -1.0;
        data[[1, 2, 0]] = 1.0;
        data[[2, 0, 1]] = 1.0;
        let v = Vector::<3>::new(data, 3, g);

        // Grade-3: rev sign = (-1)^{3*2/2} = (-1)^3 = -1
        let v_rev = v.rev();
        assert_eq!(v_rev.component(&[0, 1, 2]), -1.0);
    }
}
