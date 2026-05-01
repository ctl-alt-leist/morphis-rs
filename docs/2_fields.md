# Fields

A field is a spatially-varying geometric algebra object on a periodic grid. The abstraction extends morphis's pointwise algebra — the same metric awareness, grade structure, and product suite — to objects distributed over space, with spectral differential operators.

## Grid

`Grid<D>` defines the periodic domain geometry: a uniform cubic grid with `n_cells` points per side and total side length `box_length`. The cell length `box_length / n_cells` is derived automatically.

```rust
let grid = Grid::<3>::new(64, 100.0);
// 64^3 = 262,144 grid points, each cell 100/64 ≈ 1.5625 units wide
```

The grid also provides wavenumber calculations for spectral operations: `wavenumber(m)` returns $2\pi m / L$ with appropriate negative-frequency wrapping for indices above `N/2`. The method `cell_volume()` returns the volume of a single grid cell ($\text{cell\_length}^D$).

## Field

`Field<D>` stores a grade-k element at each grid point. The internal data layout is a single `ArrayD<f64>` with shape `[N; D] ++ [D; grade]` — spatial axes followed by tensor axes.

### Construction

```rust
// Zero field
let f = Field::zeros(grade, &grid, metric);

// From a scalar function
let rho = Field::scalar_field(&grid, g, |x| (k * x[0]).sin());

// From a vector-valued function
let v = Field::from_fn(1, &grid, g, |x| {
    // ... return a Vector<D>
});

// Constant field (same value everywhere)
let uniform = Field::constant(&some_vector, &grid);
```

### Pointwise Access

Each grid point holds a morphis `Vector<D>`:

```rust
let v: Vector<3> = field.at(&[m, n, p]);
field.set(&[m, n, p], &v);
```

### Algebra

Fields support the same operations as their pointwise elements:

- Arithmetic: `+`, `-`, negation, scalar multiplication
- `Field::wedge(&f, &g)` — pointwise exterior product (raises grade)
- `Field::interior_left(&f, &g)` — pointwise left interior product (lowers grade)
- `Field::scalar_product(&f, &g)` — pointwise scalar product
- `f.norm_squared()` — pointwise squared norm (returns scalar field)
- `f.rev()` — pointwise reversal

### Integration

For scalar fields: `f.integrate()` returns the volume integral $\int f \, dV$, and `f.sum()` returns the unweighted sum. For fields of any grade, `f.integrate_norm_squared()` returns $\int |f|^2 \, dV$ without allocating an intermediate scalar field.

### Pointwise Scaling

`Field::pointwise_scale(&scalar_field, &field)` multiplies a field of any grade by a spatially varying scalar field. This fills the gap between constant scaling (`&f * 3.0`) and full geometric products.

## Spectral Derivatives

All derivatives are computed in Fourier space with spectral accuracy on the periodic domain. The implementation uses complex-to-complex FFT via `ndrustfft`.

### Partial Derivative

`f.partial(axis)` differentiates with respect to spatial axis `a` by multiplying the Fourier coefficients by $i k_a$. Grade-preserving: operates on each tensor component independently. The Nyquist mode ($m = N/2$) is zeroed — the standard treatment for odd-order spectral derivatives on real data.

### Gradient (exterior derivative, grade-raising)

$$
\nabla f = \sum_a e_a \wedge \partial_a f
$$

`f.grad()` raises grade by 1: scalar → vector, vector → bivector, etc.

### Divergence (interior derivative, grade-lowering)

$$
\nabla \cdot f = \sum_a e_a \lrcorner \partial_a f
$$

`f.div()` lowers grade by 1: vector → scalar, bivector → vector, etc.

### Curl (exterior derivative on vectors)

`f.curl()` is the exterior derivative applied to any field — identical to `grad()`. For a vector field in 3D it produces a bivector field (the curl).

### Laplacian (grade-preserving)

$$
\nabla^2 f = \sum_a \partial^2_a f
$$

`f.laplacian()` multiplies Fourier coefficients by $-|k|^2$. Operates on each tensor component independently.

### Laplacian Inverse (Poisson solve)

`f.laplacian_inverse()` solves $\nabla^2 \phi = f$ spectrally: divides by $-|k|^2$ in Fourier space with the zero mode set to zero. Returns the unique zero-mean solution on the periodic domain.

### Identities

The spectral implementation satisfies these to machine precision:

- `f.grad().div()` = `f.laplacian()` (Hodge identity for scalar fields)
- `f.grad().curl()` = 0 (d² = 0, exterior derivative squared is zero)
- `f.laplacian_inverse().laplacian()` = `f` (for zero-mean fields)

## Even Subalgebra Field

`EvenField<D>` represents a field valued in $G^+ = G^0 \oplus G^D$ — the even subalgebra. In 3D with Euclidean metric, the pseudoscalar $I = e_1 \wedge e_2 \wedge e_3$ satisfies $I^2 = -1$, making $G^+$ isomorphic to the complex numbers. Each grid point stores $\alpha = a + bI$.

This type is the natural representation of wavefunctions in field theories: the amplitude and phase are encoded as the scalar and pseudoscalar parts.

### Algebraic Operations

- `psi.rev()` — complex conjugation: $(a + bI) \to (a - bI)$
- `psi.mul(&other)` — pointwise complex multiplication (closed in even subalgebra)
- `psi.norm_squared()` — returns scalar field $a^2 + b^2$
- `psi.rotate_phase(&angle)` — multiply by $\exp(I\theta)$ pointwise
- `psi.density(mass)` — extract $\rho = m |\alpha|^2$
- `psi.integrate_norm_squared()` — conserved norm $\int |\alpha|^2 \, dV$
- `psi.at(&indices)` — extract as `MultiVector<D>`

### Spectral Operations

- `psi.laplacian()` — spectral Laplacian applied componentwise, returns `EvenField<D>`
- `psi.gradient_components()` — returns `[grad(scalar), grad(pseudoscalar)]` as two grade-1 vector fields, with Nyquist zeroing
- `psi.kinetic_energy_density()` — gradient energy $\frac{1}{2}(|\nabla a|^2 + |\nabla b|^2)$

### Madelung Representation

The Madelung decomposition connects the wavefunction to fluid variables:

- `EvenField::madelung_inverse(&density, &velocity_potential, mass, diffusivity)` — builds $\alpha$ from $\rho$ and $\phi_v$
- `psi.madelung_velocity(diffusivity)` — extracts the velocity field $v_d = \frac{\nu}{|\alpha|^2}(a \, \partial_d b - b \, \partial_d a)$

### Phase Rotation

`rotate_phase` is the key operation for split-step integration: in Fourier space the kinetic step is a phase rotation by the dispersion relation, and in real space the potential step is a phase rotation by the potential.

## Design

The field abstraction is the mathematical substrate for field theories. It provides grids, derivatives, pointwise algebra, even-subalgebra operations, and the Laplacian inverse. Physical constants, integrator orchestration, initial conditions, and cosmological factors belong in application code.
