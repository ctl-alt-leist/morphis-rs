# Hierarchical Closure Dynamics: Context for morphis-rs

This document describes the downstream application that motivates the design of morphis-rs as a library. The application itself — a multi-scale simulation framework called Hierarchical Closure Dynamics (HCD) — will live in a separate crate. Morphis provides the core algebra; HCD provides the physics and numerics.

## The Problem

Every discipline that writes equations for averaged quantities encounters the closure problem: coarse-grained equations involve fine-scale correlations that the coarse model cannot compute. Turbulence modelers call this the Reynolds stress closure. Cosmological simulators call it subgrid physics. Kinetic theorists encounter it when truncating moment hierarchies. The mathematical structure is the same in each case.

HCD approaches this through a hierarchy of state spaces at different scales, connected by restriction (fine-to-coarse averaging) and prolongation (coarse-to-fine reconstruction) operators. The central design commitment is that these state spaces are grade-stratified: each scale $s$ carries a multivector state

$$
Ψ^{(s)} \in \mathcal{V}_s = \bigoplus_{k = 0}^{k_{\max}(s)} \mathcal{G}^k \otimes \mathcal{F}_s
$$

where $k_{\max}(s)$ is monotone — coarser scales carry fewer active grades. At cosmological scales only scalars (density, gravitational potential) are active. At galactic scales, vectors (bulk flow) and bivectors (angular momentum, magnetic field, vorticity) activate.

## Products and Elements

The algebra is built on grade-$k$ elements (k-vectors) living in a $d$-dimensional space with metric $g$. A grade-$0$ element is a scalar, grade $1$ is a vector, grade $2$ a bivector, and so on up to the pseudoscalar at grade $d$. The `Vector<D>` type represents a homogeneous k-vector; `MultiVector<D>` is a sparse sum of k-vectors at different grades.

The four products of geometric algebra are all implemented, with operator overloads that read like the math:

**Wedge (exterior) product** $u \wedge v$, written `u ^ v`. Antisymmetric, grade-additive: $\text{grade}(u \wedge v) = \text{grade}(u) + \text{grade}(v)$. This is how oriented subspaces are built — two vectors wedge to a bivector (an oriented plane), three to a trivector (an oriented volume). The implementation contracts through the generalized Kronecker delta:

$$
(u \wedge v)^{m_1 \ldots m_n} = \frac{n!}{j! \, k!} \, u^{a_1 \ldots a_j} \, v^{b_1 \ldots b_k} \, δ^{m_1 \ldots m_n}_{a_1 \ldots a_j \, b_1 \ldots b_k}
$$

**Geometric (Clifford) product** $u v$, written `u * v`. The fundamental product of the algebra. For grade-$j$ and grade-$k$ inputs, it produces components at grades $|j - k|, \, |j - k| + 2, \, \ldots, \, j + k$, each arising from a different number of metric contractions $c = (j + k - r) / 2$. For two vectors this decomposes as $u v = u \cdot v + u \wedge v$ — the sum of the inner and outer products. The result is a `MultiVector<D>`.

**Left interior product** $u \lrcorner v$, written `u << v`. Contracts all indices of $u$ into the first indices of $v$, reducing grade: $\text{grade}(u \lrcorner v) = \text{grade}(v) - \text{grade}(u)$.

**Right interior product** $u \llcorner v$, written `u >> v`. Contracts all indices of $v$ into the last indices of $u$: $\text{grade}(u \llcorner v) = \text{grade}(u) - \text{grade}(v)$.

## Transforms

The sandwich product $v \mapsto M v \tilde{M}$ is the native transformation mechanism of geometric algebra. Every isometry of the underlying space — rotation, reflection, translation (in PGA) — can be expressed as a sandwich product with an appropriate versor $M$. Rotors are plain `MultiVector` values — no special wrapper type — so the sandwich product is written as it appears in the math:

```rust
let R = rotor(&plane, angle);
let rotated = (&(&R * &v) * &R.rev()).grade_project(1);  // R v ~R
let composed = &R2 * &R1;                                 // composition
let recovered = transform(&rotated, &R.rev());             // ~R undoes R
```

A rotor is constructed from a bivector plane and an angle via the exponential map $R = e^{-B θ / 2} = \cos(θ / 2) - \hat{B} \sin(θ / 2)$. The sandwich product preserves grade — rotating a bivector returns a bivector, rotating a vector returns a vector. The `transform(v, M)` free function computes $M v \tilde{M}$ with grade projection in one step; the explicit form `R * v * R.rev()` is equally valid when the caller handles grades themselves.

Projection decomposes a vector into components parallel and perpendicular to a blade:

$$
\text{proj}_B(u) = (u \lrcorner B) \lrcorner B^{-1}
\qquad
\text{rej}_B(u) = u - \text{proj}_B(u)
$$

For grade-$1$ vectors, this reduces to the familiar $(u \cdot v) / |v|^2 \, v$. The `project` and `reject` functions are the building blocks for conservation-enforcing projection layers in learned closures. Reflection through the hyperplane perpendicular to $n$ is $\text{refl}_n(u) = -n \, u \, n^{-1}$, implemented as `reflect(u, n)`.

## Planned: Linear Maps

An outermorphism is a grade-$1$ linear map (a $d \times d$ matrix) that extends to all grades via exterior powers:

$$
(\bigwedge^k A)(B) = \underbrace{A \otimes \cdots \otimes A}_{k} \cdot B
$$

The defining property is that $A(u \wedge v) = A(u) \wedge A(v)$ — the map preserves the wedge product structure. The restriction operator $R^{(s \leftarrow s+1)}: \mathcal{V}_{s+1} \to \mathcal{V}_s$ is an outermorphism composed with grade truncation. The prolongation operator's deterministic component (spatial refinement without grade creation) is also an outermorphism.

General operators are linear maps $L: \mathcal{G}^j \to \mathcal{G}^k$ between different grade spaces. The learned closure $δQ^{(s)}$ is a general operator: it maps the state at scale $s$ to a correction term, potentially mixing grades. The operator algebra we need includes:

| Operation | Notation | Purpose in HCD |
|-----------|----------|---------------|
| Application | $L(v)$ | Applying restriction, prolongation, or closure |
| Composition | $L \circ M$ | Chaining restriction across multiple scales |
| Adjoint | $L^\dagger$ | Defining prolongation dual to restriction |
| Pseudoinverse | $L^+$ | Solving for fine-scale states from coarse data |
| SVD | $L = U Σ V^\dagger$ | Training closures via truncated decomposition |
| Determinant | $\det(A)$ | Volume scaling under outermorphisms |

## Bivector-Native Physics

A distinctive feature of the HCD framework is that it treats bivector-valued fields as first-class objects. The magnetic field is genuinely a bivector — it represents an oriented plane, the plane of the current loop that would generate it locally. Conventional MHD represents it as a pseudovector and accumulates sign-convention difficulties. Angular momentum and vorticity are also bivector-valued.

This means morphis-rs must handle bivector operations with the same fluency as vector operations: norms, inverses, and normalization; rotation via the sandwich product; interior products between vectors and bivectors; wedge products producing trivectors from vector-bivector pairs; and outermorphisms acting on bivectors via the second exterior power $\bigwedge^2 A$. The first four are implemented. The last is planned for PR 2.

## What Lives Elsewhere

The simulation framework itself — adaptive mesh hierarchy, neural network closures, training loops, conservation monitoring, zoom triggers — belongs in a separate crate that depends on morphis-rs. Morphis provides the algebra; the simulation crate provides the physics and numerics.

The boundary between the two is clean: morphis-rs knows about elements, products, linear maps, and decompositions. It does not know about grids, time integration, loss functions, or physical units.
