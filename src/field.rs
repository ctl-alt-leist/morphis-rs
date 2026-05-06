use ndarray::{ArrayD, IxDyn};

use crate::antisymmetric::n_components;
use crate::grid::Grid;
use crate::metric::Metric;
use crate::vector::Vector;

/// A field of grade-k geometric algebra elements on a periodic grid.
///
/// Each grid point holds a `Vector<D>` of the specified grade.
/// The field carries the metric and grid geometry, enabling both
/// pointwise algebraic operations and spatial differential operators.
///
/// For grade k on an N^D grid, the data shape is `[N; D] ++ [C(D, k)]`:
/// the first D axes are spatial indices (each of size n_cells) and
/// the final axis holds the C(D, k) independent tensor components.
/// For grade 0 (scalar fields), the shape is just `[N; D]`.
pub struct Field<const D: usize> {
    /// Field data with shape [N, N, ..., N, n_components].
    pub data: ArrayD<f64>,
    /// Grade of each element (0 = scalar, 1 = vector, 2 = bivector, ...).
    grade: usize,
    /// Metric defining the inner product structure.
    pub metric: Metric<D>,
    /// Grid geometry.
    pub grid: Grid<D>,
}

impl<const D: usize> Field<D> {
    /// Create a field from raw data.
    ///
    /// Data shape must be `[n_cells; D] ++ [C(D, grade)]` (or `[n_cells; D]` for scalars).
    pub fn new(data: ArrayD<f64>, grade: usize, grid: &Grid<D>, metric: Metric<D>) -> Self {
        let expected_shape = field_shape::<D>(grid.n_cells, grade);
        assert_eq!(
            data.shape(),
            expected_shape.as_slice(),
            "data shape {:?} does not match expected {:?} for grade-{} field",
            data.shape(),
            expected_shape,
            grade,
        );
        Self {
            data,
            grade,
            metric,
            grid: *grid,
        }
    }

    /// Zero field: every grid point holds a zero k-vector.
    pub fn zeros(grade: usize, grid: &Grid<D>, metric: Metric<D>) -> Self {
        let shape = field_shape::<D>(grid.n_cells, grade);
        let data = ArrayD::zeros(IxDyn(&shape));
        Self {
            data,
            grade,
            metric,
            grid: *grid,
        }
    }

    /// Constant field: every grid point holds the same value.
    pub fn constant(value: &Vector<D>, grid: &Grid<D>) -> Self {
        let grade = value.grade();
        let metric = value.metric;
        let n = grid.n_cells;
        let shape = field_shape::<D>(n, grade);
        let mut data = ArrayD::zeros(IxDyn(&shape));

        let n_comp = n_components(D, grade);

        for spatial_idx in spatial_indices_iter::<D>(n) {
            if grade == 0 {
                data[IxDyn(&spatial_idx)] = value.scalar_value();
            } else {
                let mut full_idx = spatial_idx.clone();
                full_idx.push(0); // placeholder for component index
                for c in 0..n_comp {
                    *full_idx.last_mut().unwrap() = c;
                    data[IxDyn(&full_idx)] = value.canonical_component(c);
                }
            }
        }

        Self {
            data,
            grade,
            metric,
            grid: *grid,
        }
    }

    /// Construct a field by evaluating a function at each grid point.
    pub fn from_fn(
        grade: usize,
        grid: &Grid<D>,
        metric: Metric<D>,
        f: impl Fn(&[f64; D]) -> Vector<D>,
    ) -> Self {
        let n = grid.n_cells;
        let shape = field_shape::<D>(n, grade);
        let mut data = ArrayD::zeros(IxDyn(&shape));
        let n_comp = n_components(D, grade);

        let mut indices = [0usize; D];
        for spatial_idx in spatial_indices_iter::<D>(n) {
            indices[..D].copy_from_slice(&spatial_idx[..D]);
            let pos = grid.position(&indices);
            let value = f(&pos);
            assert_eq!(value.grade(), grade, "function returned wrong grade");

            if grade == 0 {
                data[IxDyn(&spatial_idx)] = value.scalar_value();
            } else {
                let mut full_idx = spatial_idx.clone();
                full_idx.push(0);
                for c in 0..n_comp {
                    *full_idx.last_mut().unwrap() = c;
                    data[IxDyn(&full_idx)] = value.canonical_component(c);
                }
            }
        }

        Self {
            data,
            grade,
            metric,
            grid: *grid,
        }
    }

    /// Scalar field from a scalar-valued function.
    pub fn scalar_field(grid: &Grid<D>, metric: Metric<D>, f: impl Fn(&[f64; D]) -> f64) -> Self {
        let n = grid.n_cells;
        let shape = field_shape::<D>(n, 0);
        let mut data = ArrayD::zeros(IxDyn(&shape));

        let mut indices = [0usize; D];
        for spatial_idx in spatial_indices_iter::<D>(n) {
            indices[..D].copy_from_slice(&spatial_idx[..D]);
            let pos = grid.position(&indices);
            data[IxDyn(&spatial_idx)] = f(&pos);
        }

        Self {
            data,
            grade: 0,
            metric,
            grid: *grid,
        }
    }

    /// Grade of elements in this field.
    pub fn grade(&self) -> usize {
        self.grade
    }

    /// Grid geometry.
    pub fn grid(&self) -> &Grid<D> {
        &self.grid
    }

    /// Total number of grid points.
    pub fn n_points(&self) -> usize {
        self.grid.n_points()
    }

    /// Extract the value at a grid point as a `Vector<D>`.
    pub fn at(&self, indices: &[usize]) -> Vector<D> {
        assert_eq!(indices.len(), D, "expected {} spatial indices", D);
        let grade = self.grade;

        if grade == 0 {
            let val = self.data[IxDyn(indices)];
            Vector::scalar(val, self.metric)
        } else {
            let n_comp = n_components(D, grade);
            let mut full_idx: Vec<usize> = indices.to_vec();
            full_idx.push(0);
            let comp_data: Vec<f64> = (0..n_comp)
                .map(|c| {
                    *full_idx.last_mut().unwrap() = c;
                    self.data[IxDyn(&full_idx)]
                })
                .collect();

            Vector::from_sparse(comp_data, grade, self.metric)
        }
    }

    /// Set the value at a grid point from a `Vector<D>`.
    pub fn set(&mut self, indices: &[usize], value: &Vector<D>) {
        assert_eq!(indices.len(), D, "expected {} spatial indices", D);
        assert_eq!(value.grade(), self.grade, "grade mismatch");
        let grade = self.grade;

        if grade == 0 {
            self.data[IxDyn(indices)] = value.scalar_value();
        } else {
            let n_comp = n_components(D, grade);
            let mut full_idx: Vec<usize> = indices.to_vec();
            full_idx.push(0);
            for c in 0..n_comp {
                *full_idx.last_mut().unwrap() = c;
                self.data[IxDyn(&full_idx)] = value.canonical_component(c);
            }
        }
    }

    /// Whether all values are zero (within tolerance).
    pub fn is_zero(&self, tol: f64) -> bool {
        self.data.iter().all(|x| x.abs() < tol)
    }

    /// Volume integral of a scalar field: int f dV.
    ///
    /// Each cell contributes value * cell_volume. Only valid for grade-0 fields.
    pub fn integrate(&self) -> f64 {
        assert_eq!(self.grade, 0, "integrate requires a scalar field (grade 0)");
        let cell_volume = self.grid.cell_length.powi(D as i32);
        self.data.sum() * cell_volume
    }

    /// Sum of all values (no volume weighting). Only valid for grade-0 fields.
    pub fn sum(&self) -> f64 {
        assert_eq!(self.grade, 0, "sum requires a scalar field (grade 0)");
        self.data.sum()
    }

    /// Integrated L2 norm: int |f|^2 dV.
    pub fn integrate_norm_squared(&self) -> f64 {
        let n = self.grid.n_cells;
        let mut sum = 0.0;

        for spatial_idx in spatial_indices_iter::<D>(n) {
            let v = self.at(&spatial_idx);
            sum += v.norm_squared();
        }

        sum * self.grid.cell_volume()
    }

    /// Pointwise multiplication of a field by a spatially varying scalar field.
    pub fn pointwise_scale(scalar_field: &Field<D>, field: &Field<D>) -> Field<D> {
        assert_eq!(
            scalar_field.grade, 0,
            "first argument must be a scalar field (grade 0)"
        );
        assert_eq!(
            scalar_field.grid, field.grid,
            "fields must share the same grid"
        );
        let n = field.grid.n_cells;
        let grade = field.grade;
        let shape = field_shape::<D>(n, grade);
        let mut data = ArrayD::zeros(IxDyn(&shape));
        let n_comp = n_components(D, grade);

        for spatial_idx in spatial_indices_iter::<D>(n) {
            let s = scalar_field.data[IxDyn(&spatial_idx)];
            if grade == 0 {
                data[IxDyn(&spatial_idx)] = s * field.data[IxDyn(&spatial_idx)];
            } else {
                let mut full_idx = spatial_idx.clone();
                full_idx.push(0);
                for c in 0..n_comp {
                    *full_idx.last_mut().unwrap() = c;
                    data[IxDyn(&full_idx)] = s * field.data[IxDyn(&full_idx)];
                }
            }
        }

        Field {
            data,
            grade,
            metric: field.metric,
            grid: field.grid,
        }
    }

    /// Extract a single scalar component of the field as a scalar field.
    ///
    /// For a grade-k field, `tensor_indices` selects which component
    /// (e.g., [0, 1] selects the e0^e1 component of a bivector field).
    /// Handles antisymmetry: [1, 0] returns the negative of [0, 1].
    pub fn component_field(&self, tensor_indices: &[usize]) -> Field<D> {
        assert_eq!(
            tensor_indices.len(),
            self.grade,
            "wrong number of tensor indices"
        );
        let n = self.grid.n_cells;
        let shape = field_shape::<D>(n, 0);
        let mut data = ArrayD::zeros(IxDyn(&shape));

        // Determine the canonical component and sign
        let (sign, flat) = if self.grade == 0 {
            (1.0, 0)
        } else {
            match crate::antisymmetric::canonicalize(tensor_indices) {
                None => {
                    // Repeated indices: zero field
                    return Field {
                        data,
                        grade: 0,
                        metric: self.metric,
                        grid: self.grid,
                    };
                }
                Some((s, sorted)) => (s as f64, crate::antisymmetric::sorted_to_flat(&sorted)),
            }
        };

        for spatial_idx in spatial_indices_iter::<D>(n) {
            let mut full_idx = spatial_idx.clone();
            full_idx.push(flat);
            data[IxDyn(&spatial_idx)] = sign * self.data[IxDyn(&full_idx)];
        }

        Field {
            data,
            grade: 0,
            metric: self.metric,
            grid: self.grid,
        }
    }
}

impl<const D: usize> Clone for Field<D> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            grade: self.grade,
            metric: self.metric,
            grid: self.grid,
        }
    }
}

// =============================================================================
// Pointwise Algebraic Operations
// =============================================================================

impl<const D: usize> Field<D> {
    /// Pointwise reverse: applies the grade-dependent sign flip at each point.
    pub fn rev(&self) -> Self {
        let k = self.grade;
        let sign = if k < 2 || (k * (k - 1) / 2).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        Self {
            data: &self.data * sign,
            grade: self.grade,
            metric: self.metric,
            grid: self.grid,
        }
    }

    /// Pointwise norm squared: returns a scalar field (grade 0).
    pub fn norm_squared(&self) -> Field<D> {
        let n = self.grid.n_cells;
        let shape = field_shape::<D>(n, 0);
        let mut data = ArrayD::zeros(IxDyn(&shape));

        for spatial_idx in spatial_indices_iter::<D>(n) {
            let v = self.at(&spatial_idx);
            data[IxDyn(&spatial_idx)] = v.norm_squared();
        }

        Field {
            data,
            grade: 0,
            metric: self.metric,
            grid: self.grid,
        }
    }

    /// Pointwise wedge product of two fields.
    pub fn wedge(f: &Field<D>, g: &Field<D>) -> Field<D> {
        assert_eq!(f.grid, g.grid, "fields must share the same grid");
        let result_grade = f.grade + g.grade;
        let n = f.grid.n_cells;
        let shape = field_shape::<D>(n, result_grade);
        let mut data = ArrayD::zeros(IxDyn(&shape));
        let n_comp = n_components(D, result_grade);

        for spatial_idx in spatial_indices_iter::<D>(n) {
            let fv = f.at(&spatial_idx);
            let gv = g.at(&spatial_idx);
            let w = crate::ops::wedge(&fv, &gv);

            if result_grade == 0 {
                data[IxDyn(&spatial_idx)] = w.scalar_value();
            } else {
                let mut full_idx = spatial_idx.clone();
                full_idx.push(0);
                for c in 0..n_comp {
                    *full_idx.last_mut().unwrap() = c;
                    data[IxDyn(&full_idx)] = w.canonical_component(c);
                }
            }
        }

        Field {
            data,
            grade: result_grade,
            metric: f.metric,
            grid: f.grid,
        }
    }

    /// Pointwise left interior product.
    pub fn interior_left(f: &Field<D>, g: &Field<D>) -> Field<D> {
        assert_eq!(f.grid, g.grid, "fields must share the same grid");
        assert!(
            g.grade >= f.grade,
            "left interior product requires grade(g) >= grade(f)"
        );
        let result_grade = g.grade - f.grade;
        let n = f.grid.n_cells;
        let shape = field_shape::<D>(n, result_grade);
        let mut data = ArrayD::zeros(IxDyn(&shape));
        let n_comp = n_components(D, result_grade);

        for spatial_idx in spatial_indices_iter::<D>(n) {
            let fv = f.at(&spatial_idx);
            let gv = g.at(&spatial_idx);
            let w = crate::ops::interior_left(&fv, &gv);

            if result_grade == 0 {
                data[IxDyn(&spatial_idx)] = w.scalar_value();
            } else {
                let mut full_idx = spatial_idx.clone();
                full_idx.push(0);
                for c in 0..n_comp {
                    *full_idx.last_mut().unwrap() = c;
                    data[IxDyn(&full_idx)] = w.canonical_component(c);
                }
            }
        }

        Field {
            data,
            grade: result_grade,
            metric: f.metric,
            grid: f.grid,
        }
    }

    /// Pointwise scalar product: returns a scalar field.
    pub fn scalar_product(f: &Field<D>, g: &Field<D>) -> Field<D> {
        assert_eq!(f.grid, g.grid, "fields must share the same grid");
        assert_eq!(f.grade, g.grade, "scalar product requires same grade");
        let n = f.grid.n_cells;
        let shape = field_shape::<D>(n, 0);
        let mut data = ArrayD::zeros(IxDyn(&shape));

        for spatial_idx in spatial_indices_iter::<D>(n) {
            let fv = f.at(&spatial_idx);
            let gv = g.at(&spatial_idx);
            let product = crate::ops::geometric(&fv, &gv);
            data[IxDyn(&spatial_idx)] = product.scalar_part();
        }

        Field {
            data,
            grade: 0,
            metric: f.metric,
            grid: f.grid,
        }
    }
}

// =============================================================================
// Arithmetic Operators
// =============================================================================

impl<const D: usize> std::ops::Add for &Field<D> {
    type Output = Field<D>;

    fn add(self, rhs: &Field<D>) -> Field<D> {
        assert_eq!(
            self.grade, rhs.grade,
            "cannot add fields of different grade"
        );
        assert_eq!(self.grid, rhs.grid, "cannot add fields on different grids");
        Field {
            data: &self.data + &rhs.data,
            grade: self.grade,
            metric: self.metric,
            grid: self.grid,
        }
    }
}

impl<const D: usize> std::ops::Sub for &Field<D> {
    type Output = Field<D>;

    fn sub(self, rhs: &Field<D>) -> Field<D> {
        assert_eq!(
            self.grade, rhs.grade,
            "cannot subtract fields of different grade"
        );
        assert_eq!(
            self.grid, rhs.grid,
            "cannot subtract fields on different grids"
        );
        Field {
            data: &self.data - &rhs.data,
            grade: self.grade,
            metric: self.metric,
            grid: self.grid,
        }
    }
}

impl<const D: usize> std::ops::Neg for &Field<D> {
    type Output = Field<D>;

    fn neg(self) -> Field<D> {
        Field {
            data: -&self.data,
            grade: self.grade,
            metric: self.metric,
            grid: self.grid,
        }
    }
}

impl<const D: usize> std::ops::Mul<f64> for &Field<D> {
    type Output = Field<D>;

    fn mul(self, rhs: f64) -> Field<D> {
        Field {
            data: &self.data * rhs,
            grade: self.grade,
            metric: self.metric,
            grid: self.grid,
        }
    }
}

impl<const D: usize> std::ops::Mul<&Field<D>> for f64 {
    type Output = Field<D>;

    fn mul(self, rhs: &Field<D>) -> Field<D> {
        rhs * self
    }
}

// Owned variants
impl<const D: usize> std::ops::Add for Field<D> {
    type Output = Field<D>;
    fn add(self, rhs: Field<D>) -> Field<D> {
        (&self).add(&rhs)
    }
}

impl<const D: usize> std::ops::Sub for Field<D> {
    type Output = Field<D>;
    fn sub(self, rhs: Field<D>) -> Field<D> {
        (&self).sub(&rhs)
    }
}

impl<const D: usize> std::ops::Neg for Field<D> {
    type Output = Field<D>;
    fn neg(self) -> Field<D> {
        -&self
    }
}

impl<const D: usize> std::ops::Mul<f64> for Field<D> {
    type Output = Field<D>;
    fn mul(self, rhs: f64) -> Field<D> {
        (&self).mul(rhs)
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Shape of a field's data array: [n_cells; D] ++ [C(D, grade)] for grade > 0,
/// or [n_cells; D] for scalars.
pub(crate) fn field_shape<const D: usize>(n_cells: usize, grade: usize) -> Vec<usize> {
    let mut shape = vec![n_cells; D];
    if grade > 0 {
        shape.push(n_components(D, grade));
    }
    shape
}

/// Iterate over all spatial (grid) indices for an N^D grid.
pub(crate) fn spatial_indices_iter<const D: usize>(n: usize) -> Vec<Vec<usize>> {
    let total = n.pow(D as u32);
    let mut result = Vec::with_capacity(total);
    let mut current = vec![0usize; D];

    for _ in 0..total {
        result.push(current.clone());

        // Increment odometer (rightmost index fastest)
        let mut pos = D;
        while pos > 0 {
            pos -= 1;
            current[pos] += 1;
            if current[pos] < n {
                break;
            }
            current[pos] = 0;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::euclidean;

    #[test]
    fn field_shape_scalar() {
        let shape = field_shape::<3>(8, 0);
        assert_eq!(shape, vec![8, 8, 8]);
    }

    #[test]
    fn field_shape_vector() {
        // D=3, grade 1: C(3,1) = 3
        let shape = field_shape::<3>(4, 1);
        assert_eq!(shape, vec![4, 4, 4, 3]);
    }

    #[test]
    fn field_shape_bivector() {
        // D=3, grade 2: C(3,2) = 3
        let shape = field_shape::<3>(4, 2);
        assert_eq!(shape, vec![4, 4, 4, 3]);
    }

    #[test]
    fn spatial_indices_2d() {
        let indices = spatial_indices_iter::<2>(3);
        assert_eq!(indices.len(), 9); // 3^2
        assert_eq!(indices[0], vec![0, 0]);
        assert_eq!(indices[8], vec![2, 2]);
    }

    #[test]
    fn zeros_field_is_zero() {
        let g = euclidean::<3>();
        let grid = Grid::<3>::new(4, 1.0);
        let f = Field::zeros(1, &grid, g);
        assert!(f.is_zero(1e-15));
    }

    #[test]
    fn n_points_matches_grid() {
        let g = euclidean::<3>();
        let grid = Grid::<3>::new(4, 1.0);
        let f = Field::zeros(0, &grid, g);
        assert_eq!(f.n_points(), 64);
    }
}
