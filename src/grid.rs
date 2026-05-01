use std::f64::consts::PI;

/// Periodic grid geometry in D dimensions.
///
/// Represents a uniform cubic grid with periodic boundary conditions.
/// Each side has `n_cells` cells spanning `box_length` in physical units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid<const D: usize> {
    /// Number of cells per side.
    pub n_cells: usize,
    /// Box side length in physical units.
    pub box_length: f64,
    /// Cell side length (derived: box_length / n_cells).
    pub cell_length: f64,
}

impl<const D: usize> Grid<D> {
    /// Create a new periodic grid.
    ///
    /// # Arguments
    /// * `n_cells` - Number of cells per side (same in all dimensions).
    /// * `box_length` - Physical length of the box along each side.
    pub fn new(n_cells: usize, box_length: f64) -> Self {
        let cell_length = box_length / n_cells as f64;
        Self {
            n_cells,
            box_length,
            cell_length,
        }
    }

    /// Total number of grid points (n_cells^D).
    pub fn n_points(&self) -> usize {
        self.n_cells.pow(D as u32)
    }

    /// Volume of a single grid cell (cell_length^D).
    pub fn cell_volume(&self) -> f64 {
        self.cell_length.powi(D as i32)
    }

    /// Physical position of a grid point given its D-dimensional index.
    pub fn position(&self, indices: &[usize; D]) -> [f64; D] {
        let mut pos = [0.0; D];
        for a in 0..D {
            pos[a] = indices[a] as f64 * self.cell_length;
        }
        pos
    }

    /// Wavenumber along axis `a` for frequency index `m`.
    ///
    /// Returns k_a = 2π m / L for m in [0, N/2], wrapping negative
    /// frequencies for m > N/2.
    pub fn wavenumber(&self, m: usize) -> f64 {
        let n = self.n_cells;
        let freq = if m <= n / 2 {
            m as f64
        } else {
            m as f64 - n as f64
        };
        2.0 * PI * freq / self.box_length
    }

    /// Squared wavenumber magnitude |k|^2 for a D-dimensional frequency index.
    pub fn k_squared(&self, freq_indices: &[usize; D]) -> f64 {
        let mut k_sq = 0.0;
        for &m in freq_indices.iter() {
            let k_a = self.wavenumber(m);
            k_sq += k_a * k_a;
        }
        k_sq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_construction() {
        let grid = Grid::<3>::new(64, 100.0);
        assert_eq!(grid.n_cells, 64);
        assert_eq!(grid.box_length, 100.0);
        assert!((grid.cell_length - 100.0 / 64.0).abs() < 1e-15);
    }

    #[test]
    fn n_points() {
        let grid = Grid::<3>::new(4, 1.0);
        assert_eq!(grid.n_points(), 64); // 4^3

        let grid_2d = Grid::<2>::new(8, 1.0);
        assert_eq!(grid_2d.n_points(), 64); // 8^2
    }

    #[test]
    fn position_at_origin() {
        let grid = Grid::<3>::new(10, 5.0);
        let pos = grid.position(&[0, 0, 0]);
        assert_eq!(pos, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn position_at_index() {
        let grid = Grid::<3>::new(10, 5.0);
        let pos = grid.position(&[3, 7, 1]);
        let dx = 0.5; // 5.0 / 10
        assert!((pos[0] - 3.0 * dx).abs() < 1e-15);
        assert!((pos[1] - 7.0 * dx).abs() < 1e-15);
        assert!((pos[2] - 1.0 * dx).abs() < 1e-15);
    }

    #[test]
    fn wavenumber_zero_mode() {
        let grid = Grid::<3>::new(8, 1.0);
        assert!((grid.wavenumber(0)).abs() < 1e-15);
    }

    #[test]
    fn wavenumber_fundamental() {
        let grid = Grid::<3>::new(8, 1.0);
        let k1 = grid.wavenumber(1);
        assert!((k1 - 2.0 * PI).abs() < 1e-12);
    }

    #[test]
    fn wavenumber_negative_wrap() {
        let grid = Grid::<3>::new(8, 1.0);
        // Index 5 > N/2=4, so freq = 5 - 8 = -3
        let k = grid.wavenumber(5);
        let expected = 2.0 * PI * (-3.0);
        assert!((k - expected).abs() < 1e-12);
    }

    #[test]
    fn k_squared_single_mode() {
        let grid = Grid::<3>::new(8, 1.0);
        let k_sq = grid.k_squared(&[1, 0, 0]);
        let expected = (2.0 * PI).powi(2);
        assert!((k_sq - expected).abs() < 1e-10);
    }
}
