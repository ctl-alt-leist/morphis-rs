//! Sparse antisymmetric tensor storage via the combinatorial number system.
//!
//! A grade-k antisymmetric tensor in D dimensions has C(D, k) independent
//! components, indexed by strictly-increasing k-tuples from {0, ..., D-1}.
//! The combinatorial number system provides O(k) bidirectional mapping between
//! these sorted tuples and a flat index in [0, C(D, k)).

/// Binomial coefficient C(n, k) = n! / (k! (n-k)!).
///
/// Returns 0 when k > n.
pub fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    // Use the smaller of k and n-k for efficiency
    let k = k.min(n - k);
    let mut result = 1usize;
    for m in 0..k {
        result = result * (n - m) / (m + 1);
    }

    result
}

/// Number of independent components for a grade-k tensor in D dimensions.
pub fn n_components(dim: usize, grade: usize) -> usize {
    binomial(dim, grade)
}

/// Map a strictly-increasing index tuple to a flat position.
///
/// Uses the combinatorial number system:
///   flat = C(indices[0], 1) + C(indices[1], 2) + ... + C(indices[k-1], k)
///
/// The indices must be strictly increasing: indices[0] < indices[1] < ... < indices[k-1].
pub fn sorted_to_flat(indices: &[usize]) -> usize {
    let mut flat = 0;
    for (pos, &idx) in indices.iter().enumerate() {
        flat += binomial(idx, pos + 1);
    }

    flat
}

/// Map a flat position back to the strictly-increasing index tuple.
///
/// Inverse of `sorted_to_flat`. Recovers the k-tuple for a given flat index
/// within a D-dimensional grade-k space.
pub fn flat_to_sorted(flat: usize, dim: usize, grade: usize) -> Vec<usize> {
    if grade == 0 {
        return vec![];
    }

    let mut indices = vec![0usize; grade];
    let mut remaining = flat;

    // Decode from highest position downward
    let mut ceiling = dim;
    for pos in (0..grade).rev() {
        // Find the largest value v such that C(v, pos+1) <= remaining
        let rank = pos + 1;
        let mut v = ceiling - 1;
        while binomial(v, rank) > remaining {
            v -= 1;
        }
        indices[pos] = v;
        remaining -= binomial(v, rank);
        ceiling = v;
    }

    indices
}

/// Determine the sign and canonical (sorted) form of an arbitrary multi-index.
///
/// Returns `None` if the indices contain repeats (the component is zero).
/// Otherwise returns `(sign, sorted_indices)` where sign is +1 or -1.
pub fn canonicalize(indices: &[usize]) -> Option<(i32, Vec<usize>)> {
    let k = indices.len();
    if k <= 1 {
        return Some((1, indices.to_vec()));
    }

    // Sort with bubble sort to count transpositions
    let mut sorted = indices.to_vec();
    let mut sign = 1i32;

    for m in 0..k {
        for n in 0..(k - 1 - m) {
            if sorted[n] > sorted[n + 1] {
                sorted.swap(n, n + 1);
                sign = -sign;
            } else if sorted[n] == sorted[n + 1] {
                // Repeated index: antisymmetric tensor vanishes
                return None;
            }
        }
    }

    Some((sign, sorted))
}

/// Iterate over all strictly-increasing k-tuples from {0, ..., dim-1}.
///
/// Yields tuples in combinadic order (same order as flat indices 0, 1, 2, ...).
pub fn canonical_indices(dim: usize, grade: usize) -> Vec<Vec<usize>> {
    let n = n_components(dim, grade);
    let mut result = Vec::with_capacity(n);

    for flat in 0..n {
        result.push(flat_to_sorted(flat, dim, grade));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binomial_basic() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 1), 5);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(3, 4), 0);
    }

    #[test]
    fn n_components_examples() {
        // D=3, k=1: 3 components
        assert_eq!(n_components(3, 1), 3);
        // D=3, k=2: 3 bivector components
        assert_eq!(n_components(3, 2), 3);
        // D=4, k=2: 6 bivector components
        assert_eq!(n_components(4, 2), 6);
        // D=3, k=3: 1 trivector
        assert_eq!(n_components(3, 3), 1);
        // D=3, k=0: 1 scalar
        assert_eq!(n_components(3, 0), 1);
    }

    #[test]
    fn sorted_to_flat_grade_1() {
        // Grade 1 in D=3: [0] -> 0, [1] -> 1, [2] -> 2
        assert_eq!(sorted_to_flat(&[0]), 0);
        assert_eq!(sorted_to_flat(&[1]), 1);
        assert_eq!(sorted_to_flat(&[2]), 2);
    }

    #[test]
    fn sorted_to_flat_grade_2() {
        // Grade 2 in D=4: [0,1]->0, [0,2]->1, [1,2]->2, [0,3]->3, [1,3]->4, [2,3]->5
        assert_eq!(sorted_to_flat(&[0, 1]), 0);
        assert_eq!(sorted_to_flat(&[0, 2]), 1);
        assert_eq!(sorted_to_flat(&[1, 2]), 2);
        assert_eq!(sorted_to_flat(&[0, 3]), 3);
        assert_eq!(sorted_to_flat(&[1, 3]), 4);
        assert_eq!(sorted_to_flat(&[2, 3]), 5);
    }

    #[test]
    fn flat_to_sorted_roundtrip() {
        for dim in 2..=5 {
            for grade in 0..=dim {
                let n = n_components(dim, grade);
                for flat in 0..n {
                    let sorted = flat_to_sorted(flat, dim, grade);
                    assert_eq!(sorted_to_flat(&sorted), flat);
                }
            }
        }
    }

    #[test]
    fn canonicalize_sorted() {
        let (sign, sorted) = canonicalize(&[0, 1, 2]).unwrap();
        assert_eq!(sign, 1);
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn canonicalize_transposition() {
        let (sign, sorted) = canonicalize(&[1, 0]).unwrap();
        assert_eq!(sign, -1);
        assert_eq!(sorted, vec![0, 1]);
    }

    #[test]
    fn canonicalize_repeated() {
        assert!(canonicalize(&[1, 1]).is_none());
        assert!(canonicalize(&[0, 2, 0]).is_none());
    }

    #[test]
    fn canonical_indices_grade_2_dim_3() {
        let indices = canonical_indices(3, 2);
        assert_eq!(indices, vec![vec![0, 1], vec![0, 2], vec![1, 2]]);
    }

    #[test]
    fn canonical_indices_grade_2_dim_4() {
        let indices = canonical_indices(4, 2);
        assert_eq!(
            indices,
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![1, 2],
                vec![0, 3],
                vec![1, 3],
                vec![2, 3],
            ]
        );
    }
}
