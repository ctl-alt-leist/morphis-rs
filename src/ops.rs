use std::collections::HashMap;

use crate::antisymmetric::{canonical_indices, canonicalize, sorted_to_flat};
use crate::metric::Metric;
use crate::multivector::MultiVector;
use crate::vector::{Vector, factorial};

// =============================================================================
// Wedge Product
// =============================================================================

/// Wedge (exterior) product of two k-vectors.
///
/// Computes the antisymmetrized tensor product. For grade-j and grade-k
/// inputs, produces a grade-(j+k) result. Only iterates over independent
/// (canonical) components of both operands.
///
/// Returns a zero vector if j + k > D.
pub fn wedge<const D: usize>(u: &Vector<D>, v: &Vector<D>) -> Vector<D> {
    let j = u.grade();
    let k = v.grade();
    let n = j + k;

    // Grade exceeds dimension: result is zero
    if n > D {
        return Vector::<D>::zero(n, u.metric);
    }

    // Scalar ^ anything = scalar multiplication
    if j == 0 {
        let s = u.scalar_value();
        return v * s;
    }
    if k == 0 {
        let s = v.scalar_value();
        return u * s;
    }

    let mut result = Vector::<D>::zero(n, u.metric);

    let u_indices = canonical_indices(D, j);
    let v_indices = canonical_indices(D, k);

    // For each pair of canonical components, try to wedge them
    for (u_flat, u_idx) in u_indices.iter().enumerate() {
        let u_val = u.canonical_component(u_flat);
        if u_val.abs() < 1e-15 {
            continue;
        }

        for (v_flat, v_idx) in v_indices.iter().enumerate() {
            let v_val = v.canonical_component(v_flat);
            if v_val.abs() < 1e-15 {
                continue;
            }

            // Concatenate indices and canonicalize
            let mut combined: Vec<usize> = Vec::with_capacity(n);
            combined.extend_from_slice(u_idx);
            combined.extend_from_slice(v_idx);

            if let Some((sign, sorted)) = canonicalize(&combined) {
                let flat = sorted_to_flat(&sorted);
                let current = result.canonical_component(flat);
                result.set_canonical(flat, current + sign as f64 * u_val * v_val);
            }
        }
    }

    result
}

// =============================================================================
// Geometric Product
// =============================================================================

/// Geometric product of two k-vectors.
///
/// For vectors of grade j and k, produces components at grades
/// |j - k|, |j - k| + 2, ..., j + k (same parity as j + k).
///
/// Each grade r corresponds to c = (j + k - r) / 2 metric contractions.
pub fn geometric<const D: usize>(u: &Vector<D>, v: &Vector<D>) -> MultiVector<D> {
    let j = u.grade();
    let k = v.grade();
    let d = D;
    let g = &u.metric;

    let min_grade = j.abs_diff(k);
    let max_grade = (j + k).min(d);

    let mut components: HashMap<usize, Vector<D>> = HashMap::new();

    let mut r = min_grade;
    while r <= max_grade {
        let c = (j + k - r) / 2;

        let component = geometric_grade_component(u, v, r, c, g);
        if !component.is_zero(1e-15) {
            components.insert(r, component);
        }

        r += 2;
    }

    MultiVector::from_components(components, *g)
}

/// Compute the grade-r component of the geometric product.
///
/// Iterates over all multi-index combinations (using the canonical
/// components and their permutations) to correctly handle contractions.
fn geometric_grade_component<const D: usize>(
    u: &Vector<D>,
    v: &Vector<D>,
    r: usize,
    c: usize,
    g: &Metric<D>,
) -> Vector<D> {
    let j = u.grade();
    let k = v.grade();
    let d = D;

    if r == 0 && j == 0 && k == 0 {
        let s = u.scalar_value() * v.scalar_value();
        return Vector::scalar(s, *g);
    }

    let norm = geometric_normalization(j, k, c);

    // Iterate over all multi-index pairs, computing tensor values on the fly
    // from the sparse canonical representation.
    let mut result = Vector::<D>::zero(r, *g);

    for a_idx in multi_indices(d, j) {
        let u_val = tensor_value(u, &a_idx);
        if u_val.abs() < 1e-15 {
            continue;
        }

        for b_idx in multi_indices(d, k) {
            let v_val = tensor_value(v, &b_idx);
            if v_val.abs() < 1e-15 {
                continue;
            }

            // Contract: last c of u with first c of v (reversed pairing)
            let metric_factor = contract_indices_rev(&a_idx[j - c..], &b_idx[..c], g);
            if metric_factor.abs() < 1e-15 {
                continue;
            }

            // Free indices: first (j-c) of u, last (k-c) of v
            let contribution = norm * u_val * v_val * metric_factor;

            if r == 0 {
                let current = result.canonical_component(0);
                result.set_canonical(0, current + contribution);
            } else {
                let mut free: Vec<usize> = Vec::with_capacity(r);
                free.extend_from_slice(&a_idx[..j - c]);
                free.extend_from_slice(&b_idx[c..]);

                if let Some((sign, sorted)) = canonicalize(&free) {
                    let flat = sorted_to_flat(&sorted);
                    let current = result.canonical_component(flat);
                    result.set_canonical(flat, current + sign as f64 * contribution);
                }
            }
        }
    }

    result
}

/// Compute the product of metric components with reversed pairing.
///
/// Pairs u_contracted (reversed) with v_contracted:
///   g(u[c-1], v[0]) * g(u[c-2], v[1]) * ... * g(u[0], v[c-1])
fn contract_indices_rev<const D: usize>(
    u_contracted: &[usize],
    v_contracted: &[usize],
    g: &Metric<D>,
) -> f64 {
    u_contracted
        .iter()
        .rev()
        .zip(v_contracted.iter())
        .map(|(&a, &b)| g.component(a, b))
        .product()
}

/// Normalization factor for the geometric product at grade r with c contractions.
///
/// When iterating over all multi-indices (D^j x D^k) with direct canonicalization
/// of the free indices, the correct factor is 1/(c! * (j-c)! * (k-c)!).
fn geometric_normalization(j: usize, k: usize, c: usize) -> f64 {
    1.0 / (factorial(c) as f64 * factorial(j - c) as f64 * factorial(k - c) as f64)
}

// =============================================================================
// Interior Products
// =============================================================================

/// Left interior product: u . v
///
/// Contracts all j indices of u with the first j indices of v via the metric.
/// Result has grade k - j. Returns zero scalar if j > k.
pub fn interior_left<const D: usize>(u: &Vector<D>, v: &Vector<D>) -> Vector<D> {
    let j = u.grade();
    let k = v.grade();
    let g = &u.metric;

    if j > k {
        return Vector::scalar(0.0, *g);
    }

    let r = k - j;

    if j == 0 {
        let s = u.scalar_value();
        return v * s;
    }

    let d = D;
    let norm = 1.0 / factorial(j) as f64;
    let mut result = Vector::<D>::zero(r, *g);

    for a_idx in multi_indices(d, j) {
        let u_val = tensor_value(u, &a_idx);
        if u_val.abs() < 1e-15 {
            continue;
        }

        for b_idx in multi_indices(d, k) {
            let v_val = tensor_value(v, &b_idx);
            if v_val.abs() < 1e-15 {
                continue;
            }

            // Contract all j indices of u with first j of v
            let metric_factor: f64 = a_idx
                .iter()
                .zip(b_idx[..j].iter())
                .map(|(&a, &b)| g.component(a, b))
                .product();
            if metric_factor.abs() < 1e-15 {
                continue;
            }

            let contribution = norm * u_val * v_val * metric_factor;

            if r == 0 {
                let current = result.canonical_component(0);
                result.set_canonical(0, current + contribution);
            } else {
                let free = &b_idx[j..];
                if let Some((sign, sorted)) = canonicalize(free) {
                    let flat = sorted_to_flat(&sorted);
                    let current = result.canonical_component(flat);
                    result.set_canonical(flat, current + sign as f64 * contribution);
                }
            }
        }
    }

    result
}

/// Right interior product: u . v
///
/// Contracts all k indices of v with the last k indices of u via the metric.
/// Result has grade j - k. Returns zero scalar if k > j.
pub fn interior_right<const D: usize>(u: &Vector<D>, v: &Vector<D>) -> Vector<D> {
    let j = u.grade();
    let k = v.grade();
    let g = &u.metric;

    if k > j {
        return Vector::scalar(0.0, *g);
    }

    let r = j - k;

    if k == 0 {
        let s = v.scalar_value();
        return u * s;
    }

    let d = D;
    let norm = 1.0 / factorial(k) as f64;
    let mut result = Vector::<D>::zero(r, *g);

    for a_idx in multi_indices(d, j) {
        let u_val = tensor_value(u, &a_idx);
        if u_val.abs() < 1e-15 {
            continue;
        }

        for b_idx in multi_indices(d, k) {
            let v_val = tensor_value(v, &b_idx);
            if v_val.abs() < 1e-15 {
                continue;
            }

            // Contract all k indices of v with last k of u
            let metric_factor: f64 = a_idx[r..]
                .iter()
                .zip(b_idx.iter())
                .map(|(&a, &b)| g.component(a, b))
                .product();
            if metric_factor.abs() < 1e-15 {
                continue;
            }

            let contribution = norm * u_val * v_val * metric_factor;

            if r == 0 {
                let current = result.canonical_component(0);
                result.set_canonical(0, current + contribution);
            } else {
                let free = &a_idx[..r];
                if let Some((sign, sorted)) = canonicalize(free) {
                    let flat = sorted_to_flat(&sorted);
                    let current = result.canonical_component(flat);
                    result.set_canonical(flat, current + sign as f64 * contribution);
                }
            }
        }
    }

    result
}

// =============================================================================
// MultiVector Products
// =============================================================================

/// Geometric product: MultiVector * Vector.
pub fn geometric_mv_v<const D: usize>(m: &MultiVector<D>, v: &Vector<D>) -> MultiVector<D> {
    let mut result = MultiVector::zero(m.metric);

    for component in m.components().values() {
        let product = geometric(component, v);
        result = &result + &product;
    }

    result
}

/// Geometric product: Vector * MultiVector.
pub fn geometric_v_mv<const D: usize>(v: &Vector<D>, m: &MultiVector<D>) -> MultiVector<D> {
    let mut result = MultiVector::zero(m.metric);

    for component in m.components().values() {
        let product = geometric(v, component);
        result = &result + &product;
    }

    result
}

/// Geometric product: MultiVector * MultiVector.
pub fn geometric_mv_mv<const D: usize>(m: &MultiVector<D>, n: &MultiVector<D>) -> MultiVector<D> {
    let mut result = MultiVector::zero(m.metric);

    for m_component in m.components().values() {
        for n_component in n.components().values() {
            let product = geometric(m_component, n_component);
            result = &result + &product;
        }
    }

    result
}

// =============================================================================
// Projection and Reflection
// =============================================================================

/// Project vector u onto blade v.
///
/// proj_v(u) = (u . v) . v^{-1}
pub fn project<const D: usize>(u: &Vector<D>, v: &Vector<D>) -> Vector<D> {
    let contracted = interior_left(u, v);
    let v_inv = inverse(v).expect("cannot project onto zero or non-invertible blade");

    interior_left(&contracted, &v_inv)
}

/// Reject vector u from blade v: the component of u orthogonal to v.
pub fn reject<const D: usize>(u: &Vector<D>, v: &Vector<D>) -> Vector<D> {
    u - &project(u, v)
}

/// Reflect vector u through the hyperplane perpendicular to unit vector n.
///
/// refl_n(u) = -n u n^{-1}
pub fn reflect<const D: usize>(u: &Vector<D>, n: &Vector<D>) -> Vector<D> {
    assert_eq!(
        n.grade(),
        1,
        "reflect requires a grade-1 vector as mirror normal"
    );
    let n_inv = inverse(n).expect("cannot reflect through zero vector");

    let product = geometric_mv_v(&geometric(n, u), &n_inv);

    -&product.grade_project(1)
}

// =============================================================================
// Inverse
// =============================================================================

/// Inverse of a k-vector: u^{-1} = rev(u) / (u * rev(u))_0
///
/// Returns None if the vector has no inverse (zero norm squared).
pub fn inverse<const D: usize>(u: &Vector<D>) -> Option<Vector<D>> {
    let u_rev = u.rev();
    let product = geometric(u, &u_rev);

    let s = product.grade_select(0)?;
    let s_val = s.scalar_value();

    if s_val.abs() < 1e-15 {
        return None;
    }

    Some(&u_rev / s_val)
}

// =============================================================================
// Helpers
// =============================================================================

/// Compute the value of a k-vector's dense tensor at an arbitrary multi-index.
///
/// Uses the sparse canonical storage: canonicalizes the index, looks up the
/// flat position, and applies the permutation sign.
fn tensor_value<const D: usize>(v: &Vector<D>, indices: &[usize]) -> f64 {
    if v.grade() == 0 {
        return v.scalar_value();
    }

    match canonicalize(indices) {
        None => 0.0,
        Some((sign, sorted)) => {
            let flat = sorted_to_flat(&sorted);
            sign as f64 * v.canonical_component(flat)
        }
    }
}

/// Generate all multi-indices of length k with each index in [0, d).
fn multi_indices(d: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }

    let total = d.pow(k as u32);
    let mut result = Vec::with_capacity(total);
    let mut current = vec![0usize; k];

    for _ in 0..total {
        result.push(current.clone());

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

    result
}

// =============================================================================
// Product Operator Overloads
// =============================================================================

/// Implement a product operator for all four ownership combinations of Vector.
macro_rules! impl_product_op {
    ($trait:ident, $method:ident, $func:ident -> $output:ident) => {
        impl<const D: usize> std::ops::$trait for &Vector<D> {
            type Output = $output<D>;
            fn $method(self, rhs: Self) -> $output<D> {
                $func(self, rhs)
            }
        }

        impl<const D: usize> std::ops::$trait for Vector<D> {
            type Output = $output<D>;
            fn $method(self, rhs: Self) -> $output<D> {
                $func(&self, &rhs)
            }
        }

        impl<const D: usize> std::ops::$trait<Vector<D>> for &Vector<D> {
            type Output = $output<D>;
            fn $method(self, rhs: Vector<D>) -> $output<D> {
                $func(self, &rhs)
            }
        }

        impl<const D: usize> std::ops::$trait<&Vector<D>> for Vector<D> {
            type Output = $output<D>;
            fn $method(self, rhs: &Vector<D>) -> $output<D> {
                $func(&self, rhs)
            }
        }
    };
}

// Wedge product: u ^ v
impl_product_op!(BitXor, bitxor, wedge -> Vector);

// Geometric product: u * v
impl_product_op!(Mul, mul, geometric -> MultiVector);

// Left interior product: u << v
impl_product_op!(Shl, shl, interior_left -> Vector);

// Right interior product: u >> v
impl_product_op!(Shr, shr, interior_right -> Vector);

/// Implement a mixed product operator: LHS_type * RHS_type -> Output_type.
macro_rules! impl_mixed_product_op {
    ($trait:ident, $method:ident, $func:ident, $lhs:ident, $rhs:ident -> $output:ident) => {
        impl<const D: usize> std::ops::$trait<&$rhs<D>> for &$lhs<D> {
            type Output = $output<D>;
            fn $method(self, rhs: &$rhs<D>) -> $output<D> {
                $func(self, rhs)
            }
        }

        impl<const D: usize> std::ops::$trait<$rhs<D>> for $lhs<D> {
            type Output = $output<D>;
            fn $method(self, rhs: $rhs<D>) -> $output<D> {
                $func(&self, &rhs)
            }
        }

        impl<const D: usize> std::ops::$trait<$rhs<D>> for &$lhs<D> {
            type Output = $output<D>;
            fn $method(self, rhs: $rhs<D>) -> $output<D> {
                $func(self, &rhs)
            }
        }

        impl<const D: usize> std::ops::$trait<&$rhs<D>> for $lhs<D> {
            type Output = $output<D>;
            fn $method(self, rhs: &$rhs<D>) -> $output<D> {
                $func(&self, rhs)
            }
        }
    };
}

// Geometric product: MultiVector * Vector -> MultiVector
impl_mixed_product_op!(Mul, mul, geometric_mv_v, MultiVector, Vector -> MultiVector);

// Geometric product: Vector * MultiVector -> MultiVector
impl_mixed_product_op!(Mul, mul, geometric_v_mv, Vector, MultiVector -> MultiVector);

// Geometric product: MultiVector * MultiVector -> MultiVector
impl<const D: usize> std::ops::Mul for &MultiVector<D> {
    type Output = MultiVector<D>;
    fn mul(self, rhs: &MultiVector<D>) -> MultiVector<D> {
        geometric_mv_mv(self, rhs)
    }
}

impl<const D: usize> std::ops::Mul for MultiVector<D> {
    type Output = MultiVector<D>;
    fn mul(self, rhs: MultiVector<D>) -> MultiVector<D> {
        geometric_mv_mv(&self, &rhs)
    }
}

impl<const D: usize> std::ops::Mul<MultiVector<D>> for &MultiVector<D> {
    type Output = MultiVector<D>;
    fn mul(self, rhs: MultiVector<D>) -> MultiVector<D> {
        geometric_mv_mv(self, &rhs)
    }
}

impl<const D: usize> std::ops::Mul<&MultiVector<D>> for MultiVector<D> {
    type Output = MultiVector<D>;
    fn mul(self, rhs: &MultiVector<D>) -> MultiVector<D> {
        geometric_mv_mv(&self, rhs)
    }
}
