# The Metric

The metric introduces measurement into geometric algebra — lengths, angles, volumes, and the distinction between positional and directional information.

## The Metric Tensor

The **metric tensor** $g_{ab}$ emerges from the inner products of basis vectors:

$$g_{ab} = \mathbf{e}_a \cdot \mathbf{e}_b$$

This symmetric bilinear form encodes all geometric information about how vectors relate metrically:
- **Euclidean**: $g_{ab} = δ_{ab}$ (identity matrix)
- **Minkowski**: signature $(+, -, -, -)$ or $(-, +, +, +)$
- **Degenerate (PGA)**: some diagonal entries are zero

```rust
use morphis::metric::{euclidean, lorentzian, projective};

let g = euclidean::<3>();      // diag(1, 1, 1)
let h = projective::<4>();     // diag(0, 1, 1, 1)
let eta = lorentzian::<4>();   // diag(1, -1, -1, -1)

g.dim();                       // 3
g.component(0, 0);             // 1.0
g.component(0, 1);             // 0.0 (off-diagonal)
```

## Extended Inner Products

The metric extends from vectors to all grades. For grade-$k$ k-vectors:

$$(u_k \cdot v_k) = \frac{1}{k!} \, u^{m_1 \ldots m_k} v^{n_1 \ldots n_k} g_{m_1 n_1} \cdots g_{m_k n_k}$$

For bivectors specifically:

$$
\mathbf{e}_{mn} \cdot \mathbf{e}_{pq} = g_{mp} g_{nq} - g_{mq} g_{np}
= \begin{vmatrix}
g_{mp} & g_{mq} \\
g_{np} & g_{nq}
\end{vmatrix}
$$

The determinant structure generalizes to all grades:

$$(u_1 \wedge \cdots \wedge u_k) \cdot (v_1 \wedge \cdots \wedge v_k) = \det(u_a \cdot v_b)$$

## Norms and Forms

### Quadratic Form

For a grade-$k$ k-vector $b$, the **quadratic form** is the metric inner product with itself:

$$\text{form}(b) = \frac{1}{k!} \, b^{m_1 \ldots m_k} b^{n_1 \ldots n_k} g_{m_1 n_1} \cdots g_{m_k n_k}$$

The factorial prevents overcounting due to antisymmetry.

```rust
// Quadratic form (can be negative in non-Euclidean signatures)
let f = v.norm_squared();

// Norm: sqrt(|form(v)|), always non-negative
let n = v.norm();

// Unit vector (norm = 1)
let u = v.normalize();  // Returns Option<Vector>
```

### Form Properties

For Euclidean metrics:
- $\text{form}(\mathbf{e}_m) = g_{mm} = 1$
- $\text{form}(v) = \sum_m (v^m)^2$ (Pythagorean)

For non-Euclidean metrics:
- Form can be negative (spacelike/timelike in Minkowski)
- Null vectors have $\text{form}(v) = 0$

### Unit Vectors and Zero Vectors

$$\hat{b} = \frac{b}{\|b\|}$$

The `normalize()` method returns `None` for zero vectors.

## Metric as Bridge

The metric connects $V$ to $V^*$ via index raising and lowering:

**Lowering** (musical flat $\flat$):

$$\mathbf{v}^\flat = g(\mathbf{v}, \cdot) \in V^*, \quad v_m = g_{mn} v^n$$

**Raising** (musical sharp $\sharp$):

$$ω^\sharp = g^{-1}(ω, \cdot) \in V, \quad ω^m = g^{mn} ω_n$$

where $g^{mn}$ is the inverse metric satisfying $g^{mp} g_{pn} = δ^m_n$.

## Metric-Dependent Operations

These operations require the metric:

| Operation | Description |
|-----------|-------------|
| Interior product | Contraction using metric |
| Hodge dual | Complement + metric |
| Norm | Length measurement |
| Projection | Orthogonal decomposition |
| Distance | Metric distance between elements |
| Inverse | Uses norm for computation |

## Summary

The metric is the single piece of additional structure that elevates bare linear algebra to geometry. It defines inner products and lengths, connects $V$ to $V^*$, determines which rotations preserve it, specifies the Clifford relation $\mathbf{v}^2 = g(\mathbf{v}, \mathbf{v})$, and enables measurement of geometric quantities.
