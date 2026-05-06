use ndarray::{ArrayD, IxDyn};

use crate::antisymmetric::{
    binomial, canonical_indices, canonicalize, n_components, sorted_to_flat,
};
use crate::metric::Metric;

/// A k-vector in geometric algebra.
///
/// Represents a homogeneous multivector of pure grade k in a D-dimensional
/// vector space. Storage is sparse: only the C(D, k) independent components
/// (corresponding to strictly-increasing index tuples) are stored. The full
/// antisymmetric tensor semantics are maintained through the access API —
/// `component(&[1, 0])` returns `-component(&[0, 1])`.
///
/// Scalars have grade 0, vectors grade 1, bivectors grade 2, etc.
#[derive(Debug, Clone)]
pub struct Vector<const D: usize> {
    /// Independent components in combinadic order.
    /// Length is C(D, grade).
    data: Vec<f64>,
    /// Grade of this k-vector (0 = scalar, 1 = vector, 2 = bivector, ...).
    grade: usize,
    /// Metric defining the inner product structure.
    pub metric: Metric<D>,
}

impl<const D: usize> Vector<D> {
    /// Create a k-vector from a dense antisymmetric tensor (ndarray).
    ///
    /// Accepts the full [D; grade]-shaped array and extracts only the
    /// canonical (strictly-increasing) components. For grade 1, this is
    /// just a direct copy since all D components are independent.
    pub fn new(data: ArrayD<f64>, grade: usize, metric: Metric<D>) -> Self {
        if grade == 0 {
            return Self::scalar(data[IxDyn(&[])], metric);
        }

        let expected_shape: Vec<usize> = vec![D; grade];
        assert_eq!(
            data.shape(),
            expected_shape.as_slice(),
            "data shape {:?} does not match expected {:?}",
            data.shape(),
            expected_shape,
        );

        let n_comp = n_components(D, grade);
        let mut sparse = vec![0.0; n_comp];
        let canonical = canonical_indices(D, grade);

        for (flat, indices) in canonical.iter().enumerate() {
            sparse[flat] = data[IxDyn(indices)];
        }

        Self {
            data: sparse,
            grade,
            metric,
        }
    }

    /// Create a new k-vector from sparse component data.
    ///
    /// The data must have length C(D, grade), with entries ordered by
    /// the combinatorial number system (strictly-increasing index tuples).
    pub fn from_sparse(data: Vec<f64>, grade: usize, metric: Metric<D>) -> Self {
        let expected = n_components(D, grade);
        assert_eq!(
            data.len(),
            expected,
            "data length {} does not match expected C({}, {}) = {} for grade-{} vector",
            data.len(),
            D,
            grade,
            expected,
            grade,
        );

        Self {
            data,
            grade,
            metric,
        }
    }

    /// Create a zero k-vector of the given grade.
    pub fn zero(grade: usize, metric: Metric<D>) -> Self {
        let n = n_components(D, grade);

        Self {
            data: vec![0.0; n],
            grade,
            metric,
        }
    }

    /// Create a scalar (grade-0) k-vector.
    pub fn scalar(value: f64, metric: Metric<D>) -> Self {
        Self {
            data: vec![value],
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

    /// Number of independent components: C(D, grade).
    pub fn n_components(&self) -> usize {
        self.data.len()
    }

    /// Whether this k-vector is zero (all components vanish).
    pub fn is_zero(&self, tol: f64) -> bool {
        self.data.iter().all(|x| x.abs() < tol)
    }

    /// Component access for an arbitrary multi-index.
    ///
    /// Handles antisymmetry: repeated indices return 0, transposed indices
    /// return the negated canonical value.
    pub fn component(&self, indices: &[usize]) -> f64 {
        assert_eq!(
            indices.len(),
            self.grade,
            "expected {} indices for grade-{} vector, got {}",
            self.grade,
            self.grade,
            indices.len(),
        );

        if self.grade == 0 {
            return self.data[0];
        }

        match canonicalize(indices) {
            None => 0.0,
            Some((sign, sorted)) => {
                let flat = sorted_to_flat(&sorted);
                sign as f64 * self.data[flat]
            }
        }
    }

    /// Mutable access to the canonical (sorted-index) component at the given flat index.
    pub fn canonical_component(&self, flat: usize) -> f64 {
        self.data[flat]
    }

    /// Set a canonical component by flat index.
    pub fn set_canonical(&mut self, flat: usize, value: f64) {
        self.data[flat] = value;
    }

    /// Set a component at an arbitrary multi-index.
    ///
    /// The value is stored in canonical form: if the indices require
    /// a sign flip, the stored value is sign * value.
    pub fn set_component(&mut self, indices: &[usize], value: f64) {
        assert_eq!(indices.len(), self.grade);

        if self.grade == 0 {
            self.data[0] = value;
            return;
        }

        if let Some((sign, sorted)) = canonicalize(indices) {
            let flat = sorted_to_flat(&sorted);
            self.data[flat] = sign as f64 * value;
        }
        // If indices have repeats, value must be zero (ignored)
    }

    /// Access the raw sparse data slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Mutable access to the raw sparse data.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Scalar value (panics if not grade 0).
    pub fn scalar_value(&self) -> f64 {
        assert_eq!(self.grade, 0, "scalar_value requires grade 0");
        self.data[0]
    }

    /// Reverse: reverses the order of grade-1 factors.
    ///
    /// For a grade-k vector: rev(v) = (-1)^{k(k-1)/2} v
    pub fn rev(&self) -> Self {
        let k = self.grade;
        let sign = if k < 2 || (k * (k - 1) / 2).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };

        Self {
            data: self.data.iter().map(|x| x * sign).collect(),
            grade: self.grade,
            metric: self.metric,
        }
    }

    /// Squared norm: <v ~v>_0 with appropriate normalization.
    ///
    /// For a grade-k vector, sums over independent components with
    /// metric factors, accounting for the k! symmetry of the full tensor.
    pub fn norm_squared(&self) -> f64 {
        let k = self.grade;

        if k == 0 {
            let s = self.data[0];
            return s * s;
        }

        let mut sum = 0.0;
        for (flat, &val) in self.data.iter().enumerate() {
            if val.abs() < 1e-15 {
                continue;
            }

            let indices = crate::antisymmetric::flat_to_sorted(flat, D, k);
            let metric_factor: f64 = indices.iter().map(|&a| self.metric.diag[a]).product();
            sum += val * val * metric_factor;
        }

        sum
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
            data: self.data.iter().map(|x| x / n).collect(),
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

    /// Iterate over (flat_index, canonical_indices, value) for nonzero components.
    pub fn nonzero_components(&self) -> impl Iterator<Item = (usize, Vec<usize>, f64)> + '_ {
        self.data
            .iter()
            .enumerate()
            .filter_map(move |(flat, &val)| {
                if val.abs() < 1e-15 {
                    None
                } else {
                    let indices = crate::antisymmetric::flat_to_sorted(flat, D, self.grade);
                    Some((flat, indices, val))
                }
            })
    }

    /// Iterate over all (canonical_indices, value) pairs including zeros.
    pub fn all_components(&self) -> impl Iterator<Item = (Vec<usize>, f64)> + '_ {
        self.data.iter().enumerate().map(move |(flat, &val)| {
            let indices = crate::antisymmetric::flat_to_sorted(flat, D, self.grade);
            (indices, val)
        })
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
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
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
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
            grade: self.grade,
            metric: self.metric,
        }
    }
}

impl<const D: usize> std::ops::Neg for &Vector<D> {
    type Output = Vector<D>;

    fn neg(self) -> Vector<D> {
        Vector {
            data: self.data.iter().map(|x| -x).collect(),
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
            data: self.data.iter().map(|x| x * rhs).collect(),
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
            data: self.data.iter().map(|x| x / rhs).collect(),
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
        let mut data = vec![0.0; D];
        data[m] = 1.0;

        Vector::from_sparse(data, 1, metric)
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
    let mut data = vec![0.0; D];
    data[n] = 1.0;

    Vector::from_sparse(data, 1, metric)
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

// =============================================================================
// Helpers
// =============================================================================

/// Factorial of n.
pub(crate) fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Number of independent components (re-export for convenience).
pub fn n_independent(dim: usize, grade: usize) -> usize {
    binomial(dim, grade)
}

/// Iterate over all canonical (strictly-increasing) index tuples.
pub fn canonical_iter(dim: usize, grade: usize) -> Vec<Vec<usize>> {
    canonical_indices(dim, grade)
}
