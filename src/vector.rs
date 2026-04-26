use ndarray::{ArrayD, IxDyn};

use crate::metric::Metric;
use crate::util::{factorial, indices_iter};

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

    /// Inverse: u^{-1} = rev(u) / (u * rev(u))_0
    ///
    /// Returns None if the vector has no inverse (zero norm squared).
    pub fn inv(&self) -> Option<Self> {
        crate::ops::inverse(self)
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

/// Implement a binary operator for the three remaining ownership combinations,
/// delegating to the existing &T op &T implementation.
macro_rules! impl_binop_owned {
    ($trait:ident, $method:ident, $lhs:ident, $output:ident) => {
        impl<const D: usize> std::ops::$trait for $lhs<D> {
            type Output = $output<D>;
            fn $method(self, rhs: Self) -> $output<D> {
                (&self).$method(&rhs)
            }
        }

        impl<const D: usize> std::ops::$trait<$lhs<D>> for &$lhs<D> {
            type Output = $output<D>;
            fn $method(self, rhs: $lhs<D>) -> $output<D> {
                self.$method(&rhs)
            }
        }

        impl<const D: usize> std::ops::$trait<&$lhs<D>> for $lhs<D> {
            type Output = $output<D>;
            fn $method(self, rhs: &$lhs<D>) -> $output<D> {
                (&self).$method(rhs)
            }
        }
    };
}

impl_binop_owned!(Add, add, Vector, Vector);
impl_binop_owned!(Sub, sub, Vector, Vector);

impl<const D: usize> std::ops::Neg for Vector<D> {
    type Output = Vector<D>;

    fn neg(self) -> Vector<D> {
        -&self
    }
}

/// Scalar multiplication: owned vector * scalar
impl<const D: usize> std::ops::Mul<f64> for Vector<D> {
    type Output = Vector<D>;

    fn mul(self, rhs: f64) -> Vector<D> {
        &self * rhs
    }
}

/// Scalar multiplication: scalar * &vector
impl<const D: usize> std::ops::Mul<&Vector<D>> for f64 {
    type Output = Vector<D>;

    fn mul(self, rhs: &Vector<D>) -> Vector<D> {
        rhs * self
    }
}

/// Scalar multiplication: scalar * vector
impl<const D: usize> std::ops::Mul<Vector<D>> for f64 {
    type Output = Vector<D>;

    fn mul(self, rhs: Vector<D>) -> Vector<D> {
        &rhs * self
    }
}

/// Scalar division: owned vector / scalar
impl<const D: usize> std::ops::Div<f64> for Vector<D> {
    type Output = Vector<D>;

    fn div(self, rhs: f64) -> Vector<D> {
        &self / rhs
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

/// Construct a single basis vector e_n in D-dimensional space.
pub fn basis_vector<const D: usize>(n: usize, metric: Metric<D>) -> Vector<D> {
    assert!(
        n < D,
        "basis index {} out of range for {}-dimensional space",
        n,
        D
    );
    let mut data = ArrayD::zeros(IxDyn(&[D]));
    data[IxDyn(&[n])] = 1.0;

    Vector::new(data, 1, metric)
}

/// Construct a basis blade from ordered indices via wedge product.
///
/// `basis_element(&[0, 1], metric)` returns e_1 ^ e_2 (the basis bivector).
/// An empty slice returns the unit scalar.
pub fn basis_element<const D: usize>(indices: &[usize], metric: Metric<D>) -> Vector<D> {
    if indices.is_empty() {
        return Vector::scalar(1.0, metric);
    }

    let mut result = basis_vector(indices[0], metric);
    for &n in &indices[1..] {
        result = crate::ops::wedge(&result, &basis_vector(n, metric));
    }

    result
}

/// Construct the pseudoscalar: e_1 ^ e_2 ^ ... ^ e_D.
pub fn pseudoscalar<const D: usize>(metric: Metric<D>) -> Vector<D> {
    let indices: Vec<usize> = (0..D).collect();

    basis_element(&indices, metric)
}
