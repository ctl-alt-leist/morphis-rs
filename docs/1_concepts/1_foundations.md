# Mathematical Foundations

This document establishes the mathematical structures underlying geometric algebra: vector spaces, tensor products, and exterior algebra.

## Vector Spaces

We begin with a finite-dimensional real vector space $V$ of dimension $d$. The **dual space** $V^*$ consists of all linear functionals $ω: V \to \mathbb{R}$. The natural pairing between $V$ and $V^*$ is given by evaluation:

$$\langle ω, v \rangle = ω(v)$$

Given a basis $\{\mathbf{e}_m\}$ for $V$, there exists a unique **dual basis** $\{\mathbf{e}^m\}$ for $V^*$ satisfying:

$$\mathbf{e}^m(\mathbf{e}_n) = δ^m_n$$

In morphis, the dimension is a const generic parameter on the metric:

```rust
use morphis::metric::euclidean;

let g = euclidean::<3>();  // 3-dimensional Euclidean space
g.dim();                   // 3
```

## Naming Conventions

Morphis uses consistent variable naming throughout documentation and code:

| Type | Case | Default Names | Notes |
|------|------|---------------|-------|
| Vectors (any grade) | Lower | `u, v, w` | Grade-1, 2, 3, etc. all use lower |
| Blades | Lower | `b` | When emphasizing factorizability |
| Multivectors | Upper | `M, N, R, S` | Mixed-grade elements |
| Rotors | Upper | `R` | Even multivector with $R\tilde{R} = 1$ |
| Metrics | Lower | `g, h, eta` | `g` Euclidean, `h` PGA, `eta` Lorentzian |

This convention emphasizes that a bivector is still a vector (in $\bigwedge^2 V$), not a multivector.

## Tensors

A **$(p, q)$-tensor** lives in the space $V^{\otimes p} \otimes (V^*)^{\otimes q}$, with $p$ contravariant indices (upstairs) and $q$ covariant indices (downstairs). In components relative to a basis:

$$T = T^{m_1 \ldots m_p}_{n_1 \ldots n_q} \, \mathbf{e}_{m_1} \otimes \cdots \otimes \mathbf{e}_{m_p} \otimes \mathbf{e}^{n_1} \otimes \cdots \otimes \mathbf{e}^{n_q}$$

Under a change of basis $\mathbf{e}_m' = R^n_m \mathbf{e}_n$, tensor components transform to preserve the tensor itself. This transformation law is a consequence of the tensor being a basis-independent geometric object, not a definition.

## The Exterior Algebra

The **exterior algebra** $\bigwedge V$ consists of completely antisymmetric tensors:

$$\bigwedge V = \bigoplus_{k=0}^{d} \bigwedge^k V$$

The $k$-th exterior power $\bigwedge^k V$ has dimension $\binom{d}{k}$. Elements of $\bigwedge^k V$ are called **$k$-vectors** (or homogeneous multivectors of grade $k$).

The **wedge product** of vectors:

$$\mathbf{a} \wedge \mathbf{b} = \mathbf{a} \otimes \mathbf{b} - \mathbf{b} \otimes \mathbf{a}$$

In components:

$$(\mathbf{a} \wedge \mathbf{b})^{mn} = a^m b^n - a^n b^m$$

The wedge product is:
- **Anticommutative**: $\mathbf{a} \wedge \mathbf{b} = -\mathbf{b} \wedge \mathbf{a}$
- **Associative**: $(\mathbf{a} \wedge \mathbf{b}) \wedge \mathbf{c} = \mathbf{a} \wedge (\mathbf{b} \wedge \mathbf{c})$
- **Nilpotent**: $\mathbf{v} \wedge \mathbf{v} = 0$

In morphis:

```rust
use morphis::metric::euclidean;
use morphis::vector::basis;

let g = euclidean::<3>();
let [e1, e2, _] = basis(g);

// Wedge product creates a bivector
let b = e1 ^ e2;   // grade-2 vector
b.grade();          // 2
```

## Storage Convention

Morphis stores k-vectors using **full antisymmetric tensor storage**. A grade-$k$ element in $d$-dimensional space stores a tensor of shape $(d, d, \ldots, d)$ with $k$ copies of $d$.

This redundant storage enables direct tensor contraction operations without index bookkeeping and uniform grade-agnostic algorithms.

The antisymmetry constraint $b^{\ldots m \ldots n \ldots} = -b^{\ldots n \ldots m \ldots}$ is maintained by all operations.

### Antisymmetry on Components

The basis k-vectors $\mathbf{e}_{mn\ldots}$ are conventionally written as antisymmetric:

$$\mathbf{e}_{mn} = \mathbf{e}_m \wedge \mathbf{e}_n = -\mathbf{e}_{nm}$$

However, in computation we need a concrete representation. Morphis uses an ordered basis with antisymmetry carried on the components. Any k-vector can be expressed as:

$$b = b^{mn} \mathbf{e}_{mn} = \tilde{b}^{mn} \mathbf{e}^{<}_{mn}$$

where $\mathbf{e}^{<}_{mn}$ denotes the ordered basis (indices satisfy $m < n$), and the antisymmetric components $\tilde{b}^{mn}$ incorporate the full alternating structure:

$$\tilde{b}^{mn} = b^{mn} \, ε^{mn}$$

The $k$-index antisymmetric symbol $ε^{m_1 \ldots m_k}$ handles sign changes from index ordering. For a bivector in 2D:

$$b = \frac{1}{2}(u^1 v^2 - u^2 v^1) \, \mathbf{e}_{12}$$

The factor $\frac{1}{2}$ ensures proper normalization, while the antisymmetric combination arises from the Levi-Civita structure.

## Dimension Counting

The exterior algebra has total dimension $2^d$. Grade by grade:

| Grade $k$ | Dimension $\binom{d}{k}$ | Name |
|-----------|--------------------------|------|
| 0 | 1 | Scalar |
| 1 | $d$ | Vector |
| 2 | $\binom{d}{2}$ | Bivector |
| $\vdots$ | $\vdots$ | $\vdots$ |
| $d$ | 1 | Pseudoscalar |

For $d = 3$: $1 + 3 + 3 + 1 = 8 = 2^3$ total basis elements.
