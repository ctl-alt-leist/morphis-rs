# Products in Geometric Algebra

Geometric algebra provides several products that encode different geometric relationships. The **wedge product** builds higher-grade elements, the **interior product** contracts grades, and the **geometric product** unifies them all.

## The Wedge Product

The **wedge product** (exterior product) constructs higher-grade Vectors by combining lower-grade ones.

### Definition

For grade-$j$ k-vector $u$ and grade-$k$ k-vector $v$, the wedge product $u \wedge v$ is a grade-$(j + k)$ k-vector:

$$(u \wedge v)^{m_1 \ldots m_{j + k}} = \frac{1}{j! \, k!} \, u^{[m_1 \ldots m_j} v^{m_{j + 1} \ldots m_{j + k}]}$$

where brackets denote antisymmetrization.

### Properties

**Anticommutativity:**

$$u \wedge v = (-1)^{jk} \, v \wedge u$$

For grade-1 vectors:

$$\mathbf{u} \wedge \mathbf{v} = -\mathbf{v} \wedge \mathbf{u}$$

**Nilpotency:**

$$\mathbf{v} \wedge \mathbf{v} = 0$$

Linear dependence:

$$\mathbf{u} = α \mathbf{v} \implies \mathbf{u} \wedge \mathbf{v} = 0$$

**Associativity:**

$$(\mathbf{a} \wedge \mathbf{b}) \wedge \mathbf{c} = \mathbf{a} \wedge (\mathbf{b} \wedge \mathbf{c})$$

### Usage in Morphis

```rust
use morphis::metric::euclidean;
use morphis::vector::basis;

let g = euclidean::<3>();
let [e1, e2, e3] = basis(g);

let b = e1.clone() ^ e2.clone();        // Bivector
let t = e1 ^ e2 ^ e3;                   // Trivector (pseudoscalar in 3D)
```

### Geometric Interpretation

The wedge product $\mathbf{u} \wedge \mathbf{v}$ represents the **oriented parallelogram** spanned by $\mathbf{u}$ and $\mathbf{v}$. Its magnitude is the area and its orientation is determined by the order of vectors. For $k$ vectors, $\mathbf{v}_1 \wedge \cdots \wedge \mathbf{v}_k$ represents the oriented $k$-volume of the parallelepiped they span.

## The Interior Product

The **interior product** (contraction) reduces grade by contracting indices using the metric.

### Left Contraction

For grade-$j$ k-vector $u$ and grade-$k$ k-vector $v$ with $j \leq k$:

$$(u \lrcorner v)^{n_1 \ldots n_{k - j}} = u^{m_1 \ldots m_j} v_{m_1 \ldots m_j}^{\ \ \ \ \ \ \ \ n_1 \ldots n_{k - j}}$$

Result grade: $k - j$. When $j > k$: $u \lrcorner v = 0$.

### Right Contraction

$$(u \llcorner v)^{m_1 \ldots m_{j - k}} = u_{n_1 \ldots n_k}^{m_1 \ldots m_{j - k}} v^{n_1 \ldots n_k}$$

Result grade: $j - k$. When $k > j$: $u \llcorner v = 0$.

### Usage in Morphis

```rust
let b = e1.clone() ^ e2.clone();

// Left contraction: e1 ⌋ (e1 ^ e2) = e2
let v = e1 << b.clone();

// Right contraction: (e1 ^ e2) ⌊ e2 = e1
let u = b >> e2;
```

### Geometric Interpretation

The interior product $v \lrcorner b$ gives the component of $b$ "perpendicular" to $v$, with one dimension removed. It is the algebraic counterpart of orthogonal projection.

## The Dot Product

For grade-1 vectors, the **dot product** extracts the scalar (symmetric) part:

$$\mathbf{u} \cdot \mathbf{v} = g_{ab} \, u^a v^b$$

This equals the full interior product when both operands are grade-1.

## The Geometric Product

The **geometric product** is the fundamental operation of Clifford algebra, combining inner and outer products.

### For Vectors (Grade-1)

$$\mathbf{a} \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \wedge \mathbf{b}$$

The symmetric part gives the dot product (scalar), the antisymmetric part gives the wedge product (bivector).

### General Form

For general multivectors, the geometric product distributes over grades:

$$MN = \sum_{r, s} \sum_{t = |r - s|}^{r + s} \langle M_r N_s \rangle_t$$

where $M_r = \langle M \rangle_r$ and the sum over $t$ has step 2 (parity preservation).

### Usage in Morphis

```rust
let g = euclidean::<3>();
let [e1, e2, _] = basis(g);

// Orthogonal vectors: pure bivector
let m = e1.clone() * e2;
m.grades();  // vec![2]

// Parallel vectors: pure scalar
let s = e1.clone() * e1;
s.grades();  // vec![0]
```

### Properties

**Associativity:** $(MN)P = M(NP)$

**Distributivity:** $M(N + P) = MN + MP$

**Not commutative (in general):** $MN \neq NM$

### Relationship to Other Products

The wedge and interior products can be extracted from the geometric product. For k-vectors $u$ and $v$ of grades $j$ and $k$:

$$u \wedge v = \langle uv \rangle_{j + k}$$

$$u \cdot v = \langle uv \rangle_{|j - k|}$$

## The Antiwedge Product (Meet)

The **antiwedge** (regressive product, meet) finds the intersection of subspaces. For k-vectors $u$ and $v$:

$$u \vee v = \overline{\left(\overline{u} \wedge \overline{v}\right)}$$

where $\overline{\phantom{x}}$ denotes the complement.

## The Commutator and Anticommutator

The **commutator product**:

$$[M, N] = \frac{1}{2}(MN - NM)$$

The **anticommutator product**:

$$\{M, N\} = \frac{1}{2}(MN + NM)$$

The commutator of bivectors generates the Lie algebra structure of rotations.

## Scalar Product

The **scalar product** extracts only the grade-0 part of the geometric product:

$$M * N = \langle MN \rangle_0$$

## Summary Table

| Product | Symbol | Result Grade | Meaning |
|---------|--------|--------------|---------|
| Wedge | $\wedge$ / `^` | $j + k$ | Span, oriented volume |
| Interior (left) | $\lrcorner$ / `<<` | $k - j$ | Contraction |
| Interior (right) | $\llcorner$ / `>>` | $j - k$ | Contraction |
| Geometric | juxtaposition / `*` | Mixed | Full algebraic product |
| Dot | $\cdot$ | 0 | Scalar inner product |
| Antiwedge | $\vee$ | $j + k - d$ | Intersection |
| Scalar | $*$ | 0 | Grade-0 extraction |
