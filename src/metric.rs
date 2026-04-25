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
    pub fn component(&self, a: usize, b: usize) -> f64 {
        if a == b { self.diag[a] } else { 0.0 }
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
