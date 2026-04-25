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
/// Contracts the last c indices of u with the first c indices of v,
/// pairing them in reverse order as in the Python implementation.
fn contract_indices<const D: usize>(
    u_contracted: &[usize],
    v_contracted: &[usize],
    g: &Metric<D>,
) -> f64 {
    u_contracted
        .iter()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{euclidean, lorentzian};
    use crate::vector::basis;

    // =========================================================================
    // Wedge Product Tests
    // =========================================================================

    #[test]
    fn wedge_basis_vectors() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let b = wedge(&e[0], &e[1]);

        assert_eq!(b.grade(), 2);
        assert_eq!(b.component(&[0, 1]), 1.0);
        assert_eq!(b.component(&[1, 0]), -1.0);
        assert_eq!(b.component(&[0, 0]), 0.0);
    }

    #[test]
    fn wedge_anticommutativity() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let u = &(&e[0] * 2.0) + &(&e[1] * 3.0);
        let v = &(&e[1] * 1.0) + &(&e[2] * 4.0);

        let uv = wedge(&u, &v);
        let vu = wedge(&v, &u);
        let neg_vu = -&vu;

        for m in 0..3 {
            for n in 0..3 {
                assert!(
                    (uv.component(&[m, n]) - neg_vu.component(&[m, n])).abs() < 1e-12,
                    "anticommutativity failed at [{}, {}]",
                    m,
                    n,
                );
            }
        }
    }

    #[test]
    fn wedge_nilpotency() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        // e_m ^ e_m == 0 for all m
        for m in 0..3 {
            let b = wedge(&e[m], &e[m]);
            assert!(b.is_zero(1e-12), "e_{} ^ e_{} should be zero", m, m);
        }
    }

    #[test]
    fn wedge_grade_exceeds_dim() {
        let g: Metric<2> = euclidean();
        let e = basis(g);

        let b = wedge(&e[0], &e[1]);
        let trivec = wedge(&b, &e[0]);

        assert_eq!(trivec.grade(), 3);
        assert!(trivec.is_zero(1e-12));
    }

    #[test]
    fn wedge_three_basis_vectors() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let b01 = wedge(&e[0], &e[1]);
        let trivec = wedge(&b01, &e[2]);

        assert_eq!(trivec.grade(), 3);
        assert_eq!(trivec.component(&[0, 1, 2]), 1.0);
        assert_eq!(trivec.component(&[1, 0, 2]), -1.0);
        assert_eq!(trivec.component(&[0, 2, 1]), -1.0);
    }

    #[test]
    fn wedge_with_scalar() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let s = Vector::<3>::scalar(3.0, g.clone());
        let v = wedge(&s, &e[1]);

        assert_eq!(v.grade(), 1);
        assert_eq!(v.component(&[1]), 3.0);
    }

    // =========================================================================
    // Geometric Product Tests
    // =========================================================================

    #[test]
    fn geometric_basis_vectors_euclidean() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        // e_0 * e_0 = 1 (Euclidean)
        let product = geometric(&e[0], &e[0]);
        let s = product.grade_select(0).unwrap();
        assert!((s.data[IxDyn(&[])] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn geometric_product_decomposition() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        // For grade-1 vectors: u * v = (u . v) + (u ^ v)
        let u = &(&e[0] * 2.0) + &(&e[1] * 1.0);
        let v = &(&e[0] * 1.0) + &(&e[2] * 3.0);

        let product = geometric(&u, &v);

        // Scalar part = dot product = 2*1 + 1*0 + 0*3 = 2
        let scalar = product.grade_select(0).unwrap();
        assert!((scalar.data[IxDyn(&[])] - 2.0).abs() < 1e-12);

        // Bivector part = wedge product
        let bv = product.grade_select(2).unwrap();
        let w = wedge(&u, &v);
        for m in 0..3 {
            for n in 0..3 {
                assert!(
                    (bv.component(&[m, n]) - w.component(&[m, n])).abs() < 1e-12,
                    "geometric bivector != wedge at [{}, {}]",
                    m,
                    n,
                );
            }
        }
    }

    #[test]
    fn geometric_signature_lorentzian() {
        let g: Metric<4> = lorentzian();
        let e = basis(g);

        // e_0 * e_0 = +1 (timelike)
        let p0 = geometric(&e[0], &e[0]);
        let s0 = p0.grade_select(0).unwrap();
        assert!((s0.data[IxDyn(&[])] - 1.0).abs() < 1e-12);

        // e_1 * e_1 = -1 (spacelike)
        let p1 = geometric(&e[1], &e[1]);
        let s1 = p1.grade_select(0).unwrap();
        assert!((s1.data[IxDyn(&[])] + 1.0).abs() < 1e-12);
    }

    // =========================================================================
    // Interior Product Tests
    // =========================================================================

    #[test]
    fn interior_left_vector_bivector() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        // e_0 ⌋ (e_0 ^ e_1) should give e_1
        let b = wedge(&e[0], &e[1]);
        let result = interior_left(&e[0], &b);

        assert_eq!(result.grade(), 1);
        assert!((result.component(&[0]) - 0.0).abs() < 1e-12);
        assert!((result.component(&[1]) - 1.0).abs() < 1e-12);
        assert!((result.component(&[2]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn interior_right_bivector_vector() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        // (e_0 ^ e_1) ⌊ e_1 should give e_0
        let b = wedge(&e[0], &e[1]);
        let result = interior_right(&b, &e[1]);

        assert_eq!(result.grade(), 1);
        assert!((result.component(&[0]) - 1.0).abs() < 1e-12);
        assert!((result.component(&[1]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn interior_left_grade_too_high() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let b = wedge(&e[0], &e[1]);
        // bivector ⌋ vector: grade 2 > grade 1, returns zero scalar
        let result = interior_left(&b, &e[0]);

        assert_eq!(result.grade(), 0);
        assert!((result.data[IxDyn(&[])]).abs() < 1e-12);
    }

    // =========================================================================
    // Inverse Tests
    // =========================================================================

    #[test]
    fn inverse_basis_vector() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        // e_0^{-1} = e_0 (unit vector in Euclidean)
        let e0_inv = inverse(&e[0]).unwrap();
        assert_eq!(e0_inv.grade(), 1);
        assert!((e0_inv.component(&[0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn inverse_scaled_vector() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let v = &e[0] * 3.0;
        let v_inv = inverse(&v).unwrap();

        // v * v^{-1} should be scalar 1
        let product = geometric(&v, &v_inv);
        let s = product.grade_select(0).unwrap();
        assert!((s.data[IxDyn(&[])] - 1.0).abs() < 1e-12);
    }

    // =========================================================================
    // Algebraic Law Tests
    // =========================================================================

    #[test]
    fn reverse_involution() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let u = &(&e[0] * 2.0) + &(&e[1] * 3.0);
        let b = wedge(&u, &e[2]);

        // rev(rev(v)) == v for any grade
        let b_rev_rev = b.rev().rev();
        for m in 0..3 {
            for n in 0..3 {
                assert!(
                    (b.component(&[m, n]) - b_rev_rev.component(&[m, n])).abs() < 1e-12,
                    "reverse involution failed at [{}, {}]",
                    m,
                    n,
                );
            }
        }
    }

    #[test]
    fn geometric_product_grade1_decomposition() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        // For any grade-1 vectors a, b:
        //   a * b == (a . b) + (a ^ b)
        // where a . b = a ⌋ b (left interior for equal grades)
        let a = &(&e[0] * 1.0) + &(&(&e[1] * 2.0) + &(&e[2] * 3.0));
        let b = &(&e[0] * 4.0) + &(&(&e[1] * 5.0) + &(&e[2] * 6.0));

        let product = geometric(&a, &b);
        let w = wedge(&a, &b);

        // Scalar part of geometric = dot product = 1*4 + 2*5 + 3*6 = 32
        let scalar = product.grade_select(0).unwrap();
        assert!((scalar.data[IxDyn(&[])] - 32.0).abs() < 1e-12);

        // Bivector part of geometric = wedge product
        let bv = product.grade_select(2).unwrap();
        for m in 0..3 {
            for n in 0..3 {
                assert!(
                    (bv.component(&[m, n]) - w.component(&[m, n])).abs() < 1e-12,
                    "decomposition failed at [{}, {}]",
                    m,
                    n,
                );
            }
        }
    }
}
