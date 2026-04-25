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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::euclidean;
    use crate::ops::wedge;
    use crate::vector::basis;

    #[test]
    fn from_vector() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let mv = MultiVector::from_vector(e[0].clone());

        assert_eq!(mv.grades(), vec![1]);
        assert!(mv.grade_select(1).is_some());
        assert!(mv.grade_select(0).is_none());
    }

    #[test]
    fn multivector_addition() {
        let g: Metric<3> = euclidean();
        let e = basis(g.clone());

        let s = Vector::<3>::scalar(5.0, g.clone());
        let mv1 = MultiVector::from_vector(s);
        let mv2 = MultiVector::from_vector(e[0].clone());

        let sum = &mv1 + &mv2;

        assert_eq!(sum.grades(), vec![0, 1]);
    }

    #[test]
    fn multivector_reverse() {
        let g: Metric<3> = euclidean();
        let e = basis(g.clone());

        let b = wedge(&e[0], &e[1]);
        let s = Vector::<3>::scalar(1.0, g.clone());

        let mut components = HashMap::new();
        components.insert(0, s);
        components.insert(2, b);
        let mv = MultiVector::from_components(components, g);

        let mv_rev = mv.rev();

        // Scalar part unchanged
        let s_rev = mv_rev.grade_select(0).unwrap();
        assert!((s_rev.data[ndarray::IxDyn(&[])] - 1.0).abs() < 1e-12);

        // Bivector part negated (grade-2 reversal sign = -1)
        let b_rev = mv_rev.grade_select(2).unwrap();
        assert!((b_rev.component(&[0, 1]) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn is_even_rotor_like() {
        let g: Metric<3> = euclidean();
        let e = basis(g.clone());

        let s = Vector::<3>::scalar(1.0, g.clone());
        let b = wedge(&e[0], &e[1]);

        let mut components = HashMap::new();
        components.insert(0, s);
        components.insert(2, b);
        let mv = MultiVector::from_components(components, g);

        assert!(mv.is_even());
    }

    #[test]
    fn scalar_multiplication() {
        let g: Metric<3> = euclidean();
        let e = basis(g);

        let mv = MultiVector::from_vector(e[0].clone());
        let scaled = &mv * 3.0;

        let v = scaled.grade_select(1).unwrap();
        assert!((v.component(&[0]) - 3.0).abs() < 1e-12);
    }
}
