/// Metric signatures for geometric algebra.
///
/// The signature determines the diagonal of the metric tensor g_{ab},
/// which defines the inner product structure of the vector space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Signature {
    /// All diagonal entries +1: g = diag(1, 1, ..., 1)
    Euclidean,

    /// First entry +1, rest -1: g = diag(1, -1, -1, ..., -1)
    Lorentzian,

    /// First entry 0, rest +1: g = diag(0, 1, 1, ..., 1)
    Projective,
}

/// Metric tensor for a D-dimensional vector space.
///
/// Stores the diagonal metric tensor g_{ab} and its signature.
/// All supported signatures produce diagonal metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metric<const D: usize> {
    /// The diagonal entries of the metric tensor.
    pub diag: [f64; D],
    /// The algebraic signature.
    pub sig: Signature,
}

impl<const D: usize> Metric<D> {
    /// Dimension of the underlying vector space.
    pub fn dim(&self) -> usize {
        D
    }

    /// Metric tensor component g_{ab}. Off-diagonal entries are zero.
    ///
    /// Uses internal 0-based indices. For physics-convention access,
    /// translate indices via `to_internal()` first.
    pub fn component(&self, a: usize, b: usize) -> f64 {
        if a == b { self.diag[a] } else { 0.0 }
    }

    /// First valid user-facing (physics) index.
    ///
    /// Lorentzian signatures start at 0 (timelike dimension).
    /// Euclidean and projective signatures start at 1 (first spatial).
    pub fn base_index(&self) -> usize {
        match self.sig {
            Signature::Lorentzian => 0,
            Signature::Euclidean | Signature::Projective => 1,
        }
    }

    /// Last valid user-facing (physics) index.
    pub fn max_index(&self) -> usize {
        self.base_index() + D - 1
    }

    /// Convert a user-facing (physics) index to internal 0-based index.
    ///
    /// Panics with a descriptive message if the index is out of range.
    pub fn to_internal(&self, idx: usize) -> usize {
        let base = self.base_index();
        assert!(
            idx >= base && idx < base + D,
            "index {} out of range for {:?} signature (valid: {}..={})",
            idx,
            self.sig,
            base,
            self.max_index(),
        );

        idx - base
    }

    /// Convert an internal 0-based index to user-facing (physics) index.
    pub fn to_user(&self, idx: usize) -> usize {
        assert!(
            idx < D,
            "internal index {} out of range for {}-dimensional space",
            idx,
            D,
        );

        idx + self.base_index()
    }

    /// Convert a multi-index from physics convention to internal 0-based.
    pub fn to_internal_multi(&self, indices: &[usize]) -> Vec<usize> {
        indices.iter().map(|&idx| self.to_internal(idx)).collect()
    }
}

/// Euclidean metric: g = diag(1, 1, ..., 1)
pub fn euclidean<const D: usize>() -> Metric<D> {
    Metric {
        diag: [1.0; D],
        sig: Signature::Euclidean,
    }
}

/// Lorentzian metric: g = diag(1, -1, -1, ..., -1)
pub fn lorentzian<const D: usize>() -> Metric<D> {
    let mut diag = [-1.0; D];
    diag[0] = 1.0;

    Metric {
        diag,
        sig: Signature::Lorentzian,
    }
}

/// Projective (PGA) metric: g = diag(0, 1, 1, ..., 1)
pub fn projective<const D: usize>() -> Metric<D> {
    let mut diag = [1.0; D];
    diag[0] = 0.0;

    Metric {
        diag,
        sig: Signature::Projective,
    }
}
