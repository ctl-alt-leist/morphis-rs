# Objects in Geometric Algebra

This document describes the fundamental objects in geometric algebra: vectors, blades, multivectors, versors, and rotors.

## Terminology: What We Mean by "Vector"

In morphis, the term **Vector** refers to a homogeneous multivector of pure grade $k$. This is what other texts might call a "$k$-vector" or "$p$-vector". We adopt this naming because:

1. **Mathematical consistency**: A grade-$k$ element is a vector in the vector space $\bigwedge^k V$
2. **Simplicity**: Avoids the variable naming conventions ($k$-vector, $p$-vector) that differ between texts
3. **Clarity**: The grade is an explicit attribute, not embedded in the name

A `Vector` with `grade = 1` is what's traditionally called a "vector". A `Vector` with `grade = 2` is a bivector, and so on.

```rust
use morphis::metric::euclidean;
use morphis::vector::{Vector, basis};

let g = euclidean::<3>();
let [e1, e2, e3] = basis(g);

// Grade-1 vector (traditional "vector")
e1.grade();  // 1

// Grade-2 vector (bivector) via wedge operator
let b = e1 ^ e2;
b.grade();  // 2

// Grade-0 vector (scalar)
let s = Vector::<3>::scalar(1.5, g);
s.grade();  // 0
```

## Vectors ($k$-Vectors)

A **Vector** (homogeneous multivector) is an element of pure grade $k$. In components:

$$\mathbf{v}_k = v^{m_1 \ldots m_k} \mathbf{e}_{m_1 \ldots m_k}$$

where the basis $k$-vectors satisfy:

$$\mathbf{e}_{m_1 \ldots m_k} = \mathbf{e}_{m_1} \wedge \cdots \wedge \mathbf{e}_{m_k}$$

Properties of Vectors:
- **Fixed grade**: All components have the same grade
- **Antisymmetric**: $v^{\ldots m \ldots n \ldots} = -v^{\ldots n \ldots m \ldots}$
- **Form a vector space**: Can add, subtract, scalar multiply

## Blades

A **blade** (or simple $k$-vector) is a Vector that can be written as the wedge product of $k$ grade-1 vectors:

$$\mathbf{b} = \mathbf{v}_1 \wedge \mathbf{v}_2 \wedge \cdots \wedge \mathbf{v}_k$$

Blades represent **oriented $k$-dimensional subspaces**. The magnitude encodes the $k$-dimensional volume, and the orientation determines the "sense" of the subspace.

Not every k-vector is a blade. For example, in 4D:

$$\mathbf{e}_{12} + \mathbf{e}_{34}$$

is a bivector (grade-2 Vector) but cannot be factored as $\mathbf{a} \wedge \mathbf{b}$ for any grade-1 vectors $\mathbf{a}, \mathbf{b}$.

## MultiVectors

A **MultiVector** is a general element of the Clifford algebra — a sum of Vectors of different grades:

$$\mathbf{M} = \sum_{k=0}^{d} \langle \mathbf{M} \rangle_k$$

where $\langle \mathbf{M} \rangle_k$ denotes the grade-$k$ projection.

In morphis, MultiVectors are stored sparsely as a map from grade to Vector. Only nonzero grades are present:

```rust
use morphis::multivector::MultiVector;

let m = MultiVector::from_vector(e1.clone());
m.grades();         // vec![1]
m.grade_select(1);  // Some(&Vector)
m.grade_select(0);  // None
```

## Versors

A **versor** is a product of invertible grade-1 vectors using the geometric product:

$$\mathbf{V} = \mathbf{v}_1 \mathbf{v}_2 \cdots \mathbf{v}_k$$

Versors generate orthogonal transformations via the **sandwich product**:

$$\mathbf{x}' = \mathbf{V} \mathbf{x} \mathbf{V}^{-1}$$

Versors are closed under multiplication, always invertible, and their sandwich product preserves norms.

## Rotors

A **rotor** is an even versor (product of an even number of vectors) satisfying:

$$R \in \text{Cl}^+(V, g), \quad R \tilde{R} = \mathbf{1}$$

Rotors generate **rotations** (proper orthogonal transformations) via the sandwich product:

$$\mathbf{v}' = R \mathbf{v} \tilde{R}$$

The normalization $R \tilde{R} = 1$ ensures the transformation preserves norms.

## Motors (PGA)

In **Projective Geometric Algebra** (PGA), a **motor** combines rotation and translation in a single element. Motors have grades $\{0, 2\}$ and satisfy $M \tilde{M} = 1$.

$$M = R + \frac{ε}{2} \mathbf{t} R$$

where $R$ is a rotor, $\mathbf{t}$ is the translation vector, and $ε$ is the degenerate direction.

## Summary Table

| Object | Definition | Properties |
|--------|-----------|------------|
| **Vector** | Pure grade $k$ element | Homogeneous, antisymmetric |
| **Blade** | Factorizable Vector | Represents $k$-dimensional subspace |
| **MultiVector** | Sum of Vectors | General Clifford algebra element |
| **Versor** | Product of invertible vectors | Generates orthogonal transformations |
| **Rotor** | Even unit versor | Generates rotations |
| **Motor** | PGA versor | Combines rotation and translation |
