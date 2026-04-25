use std::collections::HashMap;

use ndarray::{ArrayD, IxDyn};

use crate::metric::Metric;
use crate::multivector::MultiVector;
use crate::vector::{Vector, factorial};

// =============================================================================
// Structure Constants
// =============================================================================

/// Sign of a permutation: +1 for even, -1 for odd.
fn permutation_sign(perm: &[usize]) -> i32 {
    let n = perm.len();
    let mut visited = vec![false; n];
    let mut sign = 1i32;

    for m in 0..n {
        if visited[m] {
            continue;
        }

        // Walk the cycle starting at m
        let mut cycle_len = 0;
        let mut p = m;
        while !visited[p] {
            visited[p] = true;
            p = perm[p];
            cycle_len += 1;
        }

        if cycle_len % 2 == 0 {
            sign = -sign;
        }
    }

    sign
}

/// Generalized Kronecker delta: d^{m1...mk}_{n1...nk}
///
/// Shape: [D]^{2k}. First k indices are upper, last k are lower.
///
/// d^{m1...mk}_{n1...nk} = (1/k!) sum_s sgn(s) delta^{m1}_{n_{s(1)}} ... delta^{mk}_{n_{s(k)}}
fn generalized_delta(k: usize, d: usize) -> ArrayD<f64> {
    let shape: Vec<usize> = vec![d; 2 * k];
    let mut delta = ArrayD::zeros(IxDyn(&shape));

    if k == 0 {
        return ArrayD::from_elem(IxDyn(&[]), 1.0);
    }

    let norm = 1.0 / factorial(k) as f64;

    // Iterate over all upper index combinations
    for upper in ordered_tuples(d, k) {
        // Check for repeated indices
        if has_repeats(&upper) {
            continue;
        }

        let upper_sign = permutation_sign(&sort_permutation(&upper));

        // The lower indices must be a permutation of the upper indices
        for perm in permutations(&upper) {
            let perm_sign = {
                // Find the permutation mapping sorted -> perm
                let sorted = {
                    let mut s = upper.clone();
                    s.sort();
                    s
                };
                let mapping: Vec<usize> = perm
                    .iter()
                    .map(|&p| sorted.iter().position(|&s| s == p).unwrap())
                    .collect();
                permutation_sign(&mapping)
            };

            let mut idx: Vec<usize> = Vec::with_capacity(2 * k);
            idx.extend_from_slice(&upper);
            idx.extend_from_slice(&perm);

            delta[IxDyn(&idx)] = norm * (upper_sign * perm_sign) as f64;
        }
    }

    delta
}

// =============================================================================
// Wedge Product
// =============================================================================

/// Wedge (exterior) product of two k-vectors.
///
/// Computes antisymmetrization via the generalized Kronecker delta.
/// For grade-j and grade-k inputs with n = j + k:
///
/// ```text
/// (u ^ v)^{m1...mn} = norm * u^{a1...aj} v^{b1...bk} delta^{m1...mn}_{a1...aj b1...bk}
/// ```
///
/// where norm = n! / (j! k!).
///
/// Returns a k-vector of grade j + k, or zero if j + k > D.
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
        let s = u.data[IxDyn(&[])];

        return Vector::new(&v.data * s, k, v.metric);
    }
    if k == 0 {
        let s = v.data[IxDyn(&[])];

        return Vector::new(&u.data * s, j, u.metric);
    }

    let delta = generalized_delta(n, D);
    let norm = factorial(n) as f64 / (factorial(j) as f64 * factorial(k) as f64);

    let mut result = ArrayD::zeros(IxDyn(&vec![D; n]));

    // Contract: result^{m1...mn} = norm * sum_{a,b} u^{a1...aj} v^{b1...bk} delta^{m1...mn}_{a1...aj b1...bk}
    for m_idx in indices_iter(D, n) {
        let mut sum = 0.0;

        for a_idx in indices_iter(D, j) {
            let u_val = u.data[IxDyn(&a_idx)];
            if u_val.abs() < 1e-15 {
                continue;
            }

            for b_idx in indices_iter(D, k) {
                let v_val = v.data[IxDyn(&b_idx)];
                if v_val.abs() < 1e-15 {
                    continue;
                }

                // Build delta index: [m1...mn, a1...aj, b1...bk]
                let mut delta_idx: Vec<usize> = Vec::with_capacity(2 * n);
                delta_idx.extend_from_slice(&m_idx);
                delta_idx.extend_from_slice(&a_idx);
                delta_idx.extend_from_slice(&b_idx);

                let d_val = delta[IxDyn(&delta_idx)];
                sum += u_val * v_val * d_val;
            }
        }

        result[IxDyn(&m_idx)] = norm * sum;
    }

    Vector::new(result, n, u.metric)
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

/// Compute the grade-r component of the geometric product of two k-vectors.
///
/// Uses c metric contractions on the last c indices of u with the first c
/// indices of v (in reverse pairing order), then antisymmetrizes the
/// remaining free indices via the generalized delta.
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
        // Scalar * scalar
        let s = u.data[IxDyn(&[])] * v.data[IxDyn(&[])];

        return Vector::scalar(s, *g);
    }

    let norm = geometric_normalization(j, k, c);

    if r == 0 {
        // Full contraction to scalar
        let mut sum = 0.0;

        for a_idx in indices_iter(d, j) {
            let u_val = u.data[IxDyn(&a_idx)];
            if u_val.abs() < 1e-15 {
                continue;
            }

            for b_idx in indices_iter(d, k) {
                let v_val = v.data[IxDyn(&b_idx)];
                if v_val.abs() < 1e-15 {
                    continue;
                }

                // Contract last c of u with first c of v via metric
                let metric_factor = contract_indices(&a_idx[j - c..], &b_idx[..c], g);
                sum += u_val * v_val * metric_factor;
            }
        }

        return Vector::scalar(norm * sum, *g);
    }

    if c == 0 {
        // Pure wedge (no contractions) — already has its own normalization
        let wedge_norm = factorial(r) as f64 / (factorial(j) as f64 * factorial(k) as f64);
        let scale = norm / wedge_norm;

        return &wedge(u, v) * scale;
    }

    // Mixed: c contractions + antisymmetrization of r free indices
    let delta = generalized_delta(r, d);
    let mut result = ArrayD::zeros(IxDyn(&vec![d; r]));

    for m_idx in indices_iter(d, r) {
        let mut sum = 0.0;

        for a_idx in indices_iter(d, j) {
            let u_val = u.data[IxDyn(&a_idx)];
            if u_val.abs() < 1e-15 {
                continue;
            }

            for b_idx in indices_iter(d, k) {
                let v_val = v.data[IxDyn(&b_idx)];
                if v_val.abs() < 1e-15 {
                    continue;
                }

                // Contract last c indices of u with first c indices of v
                let metric_factor = contract_indices(&a_idx[j - c..], &b_idx[..c], g);
                if metric_factor.abs() < 1e-15 {
                    continue;
                }

                // Free indices: first (j-c) of u and last (k-c) of v
                let mut free: Vec<usize> = Vec::with_capacity(r);
                free.extend_from_slice(&a_idx[..j - c]);
                free.extend_from_slice(&b_idx[c..]);

                // Delta contraction
                let mut delta_idx: Vec<usize> = Vec::with_capacity(2 * r);
                delta_idx.extend_from_slice(&m_idx);
                delta_idx.extend_from_slice(&free);

                let d_val = delta[IxDyn(&delta_idx)];
                sum += u_val * v_val * metric_factor * d_val;
            }
        }

        result[IxDyn(&m_idx)] = norm * sum;
    }

    Vector::new(result, r, *g)
}

/// Compute the product of metric components for contracted index pairs.
///
/// Pairs the last c indices of u with the first c indices of v in
/// reverse order: g(u_{last}, v_{first}) g(u_{last-1}, v_{second}) ...
///
/// The reverse pairing produces the correct signs for the Clifford product
/// when c >= 2 (e.g. the scalar part of e12 * e12 = -1, not +1).
fn contract_indices<const D: usize>(
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
fn geometric_normalization(j: usize, k: usize, c: usize) -> f64 {
    let r = j + k - 2 * c;

    if c == 0 {
        // Pure wedge
        factorial(r) as f64 / (factorial(j) as f64 * factorial(k) as f64)
    } else if r == 0 {
        // Full contraction to scalar
        1.0 / factorial(c) as f64
    } else {
        // Partial contraction
        factorial(r) as f64
            / (factorial(c) as f64 * factorial(j - c) as f64 * factorial(k - c) as f64)
    }
}

// =============================================================================
// Interior Products
// =============================================================================

/// Left interior product: u ⌋ v
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
        let s = u.data[IxDyn(&[])];

        return Vector::new(&v.data * s, k, *g);
    }

    let mut result = if r == 0 {
        ArrayD::from_elem(IxDyn(&[]), 0.0)
    } else {
        ArrayD::zeros(IxDyn(&vec![D; r]))
    };

    let norm = 1.0 / factorial(j) as f64;

    if r == 0 {
        // Full contraction to scalar
        let mut sum = 0.0;
        for a_idx in indices_iter(D, j) {
            let u_val = u.data[IxDyn(&a_idx)];
            if u_val.abs() < 1e-15 {
                continue;
            }

            for b_idx in indices_iter(D, k) {
                let v_val = v.data[IxDyn(&b_idx)];
                if v_val.abs() < 1e-15 {
                    continue;
                }

                let metric_factor = contract_indices(&a_idx, &b_idx[..j], g);
                sum += u_val * v_val * metric_factor;
            }
        }
        result[IxDyn(&[])] = norm * sum;
    } else {
        // Partial contraction
        for out_idx in indices_iter(D, r) {
            let mut sum = 0.0;
            for a_idx in indices_iter(D, j) {
                let u_val = u.data[IxDyn(&a_idx)];
                if u_val.abs() < 1e-15 {
                    continue;
                }

                // v indices: first j contracted with u, remaining r are output
                let mut v_full: Vec<usize> = Vec::with_capacity(k);
                v_full.extend_from_slice(&a_idx);
                v_full.extend_from_slice(&out_idx);

                let v_val = v.data[IxDyn(&v_full)];
                if v_val.abs() < 1e-15 {
                    continue;
                }

                let metric_factor: f64 = a_idx.iter().map(|&a| g.component(a, a)).product();
                sum += u_val * v_val * metric_factor;
            }
            result[IxDyn(&out_idx)] = norm * sum;
        }
    }

    Vector::new(result, r, *g)
}

/// Right interior product: u ⌊ v
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
        let s = v.data[IxDyn(&[])];

        return Vector::new(&u.data * s, j, *g);
    }

    let mut result = if r == 0 {
        ArrayD::from_elem(IxDyn(&[]), 0.0)
    } else {
        ArrayD::zeros(IxDyn(&vec![D; r]))
    };

    let norm = 1.0 / factorial(k) as f64;

    if r == 0 {
        // Full contraction to scalar
        let mut sum = 0.0;
        for a_idx in indices_iter(D, j) {
            let u_val = u.data[IxDyn(&a_idx)];
            if u_val.abs() < 1e-15 {
                continue;
            }

            for b_idx in indices_iter(D, k) {
                let v_val = v.data[IxDyn(&b_idx)];
                if v_val.abs() < 1e-15 {
                    continue;
                }

                let metric_factor = contract_indices(&a_idx[r..], &b_idx, g);
                sum += u_val * v_val * metric_factor;
            }
        }
        result[IxDyn(&[])] = norm * sum;
    } else {
        // Partial contraction
        for out_idx in indices_iter(D, r) {
            let mut sum = 0.0;
            for b_idx in indices_iter(D, k) {
                let v_val = v.data[IxDyn(&b_idx)];
                if v_val.abs() < 1e-15 {
                    continue;
                }

                // u indices: first r are output, last k contracted with v
                let mut u_full: Vec<usize> = Vec::with_capacity(j);
                u_full.extend_from_slice(&out_idx);
                u_full.extend_from_slice(&b_idx);

                let u_val = u.data[IxDyn(&u_full)];
                if u_val.abs() < 1e-15 {
                    continue;
                }

                let metric_factor: f64 = b_idx.iter().map(|&b| g.component(b, b)).product();
                sum += u_val * v_val * metric_factor;
            }
            result[IxDyn(&out_idx)] = norm * sum;
        }
    }

    Vector::new(result, r, *g)
}

// =============================================================================
// MultiVector Products
// =============================================================================

/// Geometric product: MultiVector * Vector.
///
/// Distributes the product over the grade components of the multivector.
pub fn geometric_mv_v<const D: usize>(m: &MultiVector<D>, v: &Vector<D>) -> MultiVector<D> {
    let mut result = MultiVector::zero(m.metric);

    for component in m.components().values() {
        let product = geometric(component, v);
        result = &result + &product;
    }

    result
}

/// Geometric product: Vector * MultiVector.
///
/// Distributes the product over the grade components of the multivector.
pub fn geometric_v_mv<const D: usize>(v: &Vector<D>, m: &MultiVector<D>) -> MultiVector<D> {
    let mut result = MultiVector::zero(m.metric);

    for component in m.components().values() {
        let product = geometric(v, component);
        result = &result + &product;
    }

    result
}

/// Geometric product: MultiVector * MultiVector.
///
/// Distributes over the grade components of both operands.
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
/// proj_v(u) = (u ⌋ v) ⌋ v^{-1}
///
/// For grade-1 vectors, this reduces to the familiar (u · v) / |v|² v.
/// For projecting a vector onto a higher-grade blade, returns the component
/// of u lying in the subspace spanned by v.
pub fn project<const D: usize>(u: &Vector<D>, v: &Vector<D>) -> Vector<D> {
    let contracted = interior_left(u, v);
    let v_inv = inverse(v).expect("cannot project onto zero or non-invertible blade");

    // contracted ⌋ v_inv gives back a grade-1 result
    interior_left(&contracted, &v_inv)
}

/// Reject vector u from blade v: the component of u orthogonal to v.
///
/// rej_v(u) = u - proj_v(u)
pub fn reject<const D: usize>(u: &Vector<D>, v: &Vector<D>) -> Vector<D> {
    u - &project(u, v)
}

/// Reflect vector u through the hyperplane perpendicular to unit vector n.
///
/// refl_n(u) = -n u n^{-1}
///
/// Flips the component of u parallel to n, preserves the perpendicular part.
pub fn reflect<const D: usize>(u: &Vector<D>, n: &Vector<D>) -> Vector<D> {
    assert_eq!(
        n.grade(),
        1,
        "reflect requires a grade-1 vector as mirror normal"
    );
    let n_inv = inverse(n).expect("cannot reflect through zero vector");

    // -n * u * n^{-1}, extract grade 1
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
    let s_val = s.data[IxDyn(&[])];

    if s_val.abs() < 1e-15 {
        return None;
    }

    Some(&u_rev / s_val)
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

// =============================================================================
// Helpers
// =============================================================================

/// Iterate over all multi-indices of length k with each index in [0, d).
fn indices_iter(d: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }

    let mut result = Vec::new();
    let mut current = vec![0usize; k];
    loop {
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
}

/// All ordered k-tuples from [0, d) (no requirement for increasing order).
fn ordered_tuples(d: usize, k: usize) -> Vec<Vec<usize>> {
    indices_iter(d, k)
}

/// Check if a slice has repeated values.
fn has_repeats(v: &[usize]) -> bool {
    for m in 0..v.len() {
        for n in (m + 1)..v.len() {
            if v[m] == v[n] {
                return true;
            }
        }
    }

    false
}

/// Find the permutation that sorts the given slice.
fn sort_permutation(v: &[usize]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..v.len()).collect();
    indices.sort_by_key(|&m| v[m]);

    indices
}

/// Generate all permutations of the given slice.
fn permutations(v: &[usize]) -> Vec<Vec<usize>> {
    let n = v.len();
    if n == 0 {
        return vec![vec![]];
    }
    if n == 1 {
        return vec![vec![v[0]]];
    }

    let mut result = Vec::new();
    for m in 0..n {
        let rest: Vec<usize> = v
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != m)
            .map(|(_, &x)| x)
            .collect();
        for mut perm in permutations(&rest) {
            perm.insert(0, v[m]);
            result.push(perm);
        }
    }

    result
}
