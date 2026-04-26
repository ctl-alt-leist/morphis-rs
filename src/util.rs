/// Iterate over all multi-indices of length `k` with each index in [0, d).
pub(crate) fn indices_iter(d: usize, k: usize) -> Vec<Vec<usize>> {
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

/// Flatten a multi-index [m0, m1, ..., mk] into a single linear index.
///
/// Uses row-major (C-order) layout: m0 * d^(k-1) + m1 * d^(k-2) + ... + mk.
pub(crate) fn flatten_index(multi_idx: &[usize], d: usize) -> usize {
    let mut flat = 0;
    for &m in multi_idx {
        flat = flat * d + m;
    }

    flat
}
