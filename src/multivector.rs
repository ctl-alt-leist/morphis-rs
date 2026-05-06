use std::collections::HashMap;

use crate::metric::Metric;
use crate::vector::Vector;

/// A multivector in geometric algebra.
///
/// Sparse representation: stores only nonzero grade components as a
/// dictionary mapping grade to Vector. Typical multivectors (rotors,
/// motors) occupy only a few grades.
#[derive(Debug, Clone)]
pub struct MultiVector<const D: usize> {
    /// Grade-indexed components. Each value is a homogeneous k-vector.
    components: HashMap<usize, Vector<D>>,
    /// Metric defining the inner product structure.
    pub metric: Metric<D>,
}

impl<const D: usize> MultiVector<D> {
    /// Create a multivector from a map of grade -> Vector.
    pub fn from_components(components: HashMap<usize, Vector<D>>, metric: Metric<D>) -> Self {
        Self { components, metric }
    }

    /// Create a multivector from a single homogeneous k-vector.
    pub fn from_vector(v: Vector<D>) -> Self {
        let metric = v.metric;
        let grade = v.grade();
        let mut components = HashMap::new();
        components.insert(grade, v);

        Self { components, metric }
    }

    /// Create a zero multivector.
    pub fn zero(metric: Metric<D>) -> Self {
        Self {
            components: HashMap::new(),
            metric,
        }
    }

    /// Grades present in this multivector, sorted.
    pub fn grades(&self) -> Vec<usize> {
        let mut g: Vec<usize> = self.components.keys().copied().collect();
        g.sort();

        g
    }

    /// Extract the grade-k component, if present.
    pub fn grade_select(&self, k: usize) -> Option<&Vector<D>> {
        self.components.get(&k)
    }

    /// Whether all present grades are even.
    pub fn is_even(&self) -> bool {
        self.components.keys().all(|k| k % 2 == 0)
    }

    /// Dimension of the underlying vector space.
    pub fn dim(&self) -> usize {
        D
    }

    /// Reverse: reverse each component.
    pub fn rev(&self) -> Self {
        let components = self.components.iter().map(|(&k, v)| (k, v.rev())).collect();

        Self {
            components,
            metric: self.metric,
        }
    }

    /// Whether all components are zero within tolerance.
    pub fn is_zero(&self, tol: f64) -> bool {
        self.components.values().all(|v| v.is_zero(tol))
    }

    /// Access the grade-indexed component map.
    pub fn components(&self) -> &HashMap<usize, Vector<D>> {
        &self.components
    }

    /// Extract grade-k component, returning a zero vector if absent.
    pub fn grade_project(&self, k: usize) -> Vector<D> {
        self.components
            .get(&k)
            .cloned()
            .unwrap_or_else(|| Vector::zero(k, self.metric))
    }

    /// Extract the scalar (grade-0) value, defaulting to 0.
    pub fn scalar_part(&self) -> f64 {
        self.components
            .get(&0)
            .map(|v| v.scalar_value())
            .unwrap_or(0.0)
    }
}

// =============================================================================
// Arithmetic Operators
// =============================================================================

impl<const D: usize> std::ops::Add for &MultiVector<D> {
    type Output = MultiVector<D>;

    fn add(self, rhs: &MultiVector<D>) -> MultiVector<D> {
        let mut components = self.components.clone();

        for (&k, v) in &rhs.components {
            components
                .entry(k)
                .and_modify(|existing| *existing = &*existing + v)
                .or_insert_with(|| v.clone());
        }

        MultiVector {
            components,
            metric: self.metric,
        }
    }
}

impl<const D: usize> std::ops::Sub for &MultiVector<D> {
    type Output = MultiVector<D>;

    fn sub(self, rhs: &MultiVector<D>) -> MultiVector<D> {
        let mut components = self.components.clone();

        for (&k, v) in &rhs.components {
            components
                .entry(k)
                .and_modify(|existing| *existing = &*existing - v)
                .or_insert_with(|| -v);
        }

        MultiVector {
            components,
            metric: self.metric,
        }
    }
}

impl<const D: usize> std::ops::Neg for &MultiVector<D> {
    type Output = MultiVector<D>;

    fn neg(self) -> MultiVector<D> {
        let components = self.components.iter().map(|(&k, v)| (k, -v)).collect();

        MultiVector {
            components,
            metric: self.metric,
        }
    }
}

impl<const D: usize> std::ops::Mul<f64> for &MultiVector<D> {
    type Output = MultiVector<D>;

    fn mul(self, rhs: f64) -> MultiVector<D> {
        let components = self.components.iter().map(|(&k, v)| (k, v * rhs)).collect();

        MultiVector {
            components,
            metric: self.metric,
        }
    }
}

impl<const D: usize> std::ops::Add for MultiVector<D> {
    type Output = MultiVector<D>;

    fn add(self, rhs: Self) -> MultiVector<D> {
        &self + &rhs
    }
}

impl<const D: usize> std::ops::Add<MultiVector<D>> for &MultiVector<D> {
    type Output = MultiVector<D>;

    fn add(self, rhs: MultiVector<D>) -> MultiVector<D> {
        self + &rhs
    }
}

impl<const D: usize> std::ops::Add<&MultiVector<D>> for MultiVector<D> {
    type Output = MultiVector<D>;

    fn add(self, rhs: &MultiVector<D>) -> MultiVector<D> {
        &self + rhs
    }
}

impl<const D: usize> std::ops::Sub for MultiVector<D> {
    type Output = MultiVector<D>;

    fn sub(self, rhs: Self) -> MultiVector<D> {
        &self - &rhs
    }
}

impl<const D: usize> std::ops::Sub<MultiVector<D>> for &MultiVector<D> {
    type Output = MultiVector<D>;

    fn sub(self, rhs: MultiVector<D>) -> MultiVector<D> {
        self - &rhs
    }
}

impl<const D: usize> std::ops::Sub<&MultiVector<D>> for MultiVector<D> {
    type Output = MultiVector<D>;

    fn sub(self, rhs: &MultiVector<D>) -> MultiVector<D> {
        &self - rhs
    }
}

impl<const D: usize> std::ops::Neg for MultiVector<D> {
    type Output = MultiVector<D>;

    fn neg(self) -> MultiVector<D> {
        -&self
    }
}

impl<const D: usize> std::ops::Mul<f64> for MultiVector<D> {
    type Output = MultiVector<D>;

    fn mul(self, rhs: f64) -> MultiVector<D> {
        &self * rhs
    }
}

impl<const D: usize> std::ops::Mul<&MultiVector<D>> for f64 {
    type Output = MultiVector<D>;

    fn mul(self, rhs: &MultiVector<D>) -> MultiVector<D> {
        rhs * self
    }
}

impl<const D: usize> std::ops::Mul<MultiVector<D>> for f64 {
    type Output = MultiVector<D>;

    fn mul(self, rhs: MultiVector<D>) -> MultiVector<D> {
        &rhs * self
    }
}
