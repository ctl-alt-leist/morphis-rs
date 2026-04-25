# Morphis

A Rust implementation of geometric algebra, ported from the Python [morphis](https://github.com/ctl-alt-leist/morphis) library. The name derives from Greek *morphe* (form) — embodying the transformation and adaptation of geometric structures across different contexts while preserving their essential nature.

## Features

- **Geometric Algebra Core**: Vectors (k-vectors), multivectors, and operations (wedge, geometric product, interior products)
- **Metric-Aware**: Objects carry their metric context (Euclidean, Lorentzian, Projective)
- **Full Tensor Storage**: Antisymmetric tensor representation enabling direct contraction operations
- **Operator Overloading**: Write geometric algebra as it reads on paper

## Quick Start

```rust
use morphis::metric::euclidean;
use morphis::vector::{basis, basis_element, pseudoscalar};

let g = euclidean::<3>();
let [e1, e2, e3] = basis(g);

// Wedge product builds blades
let e12 = e1.clone() ^ e2.clone();   // bivector
let e123 = e1 ^ e2 ^ e3;             // pseudoscalar

// Geometric product decomposes into scalar + bivector
let m = e1 * e2;                      // MultiVector with grades {0, 2}

// Interior products contract grades
let v = e1 << e12;                    // e1 ⌋ (e1 ^ e2) = e2
let u = e12 >> e2;                    // (e1 ^ e2) ⌊ e2 = e1
```

### Operator Table

| Operator | Meaning | Example |
|----------|---------|---------|
| `^` | Wedge (exterior) product | `e1 ^ e2` |
| `*` | Geometric (Clifford) product | `e1 * e2` |
| `<<` | Left interior product | `u << b` |
| `>>` | Right interior product | `b >> v` |
| `+`, `-`, `-v` | Addition, subtraction, negation | `u + v` |
| `* f64`, `/ f64` | Scalar multiply/divide | `3.0 * v` |

## Documentation

- [Project Overview](docs/0_project-overview.md) — Vision and scope
- [Concepts](docs/1_concepts/) — Mathematical foundations (vectors, products, duality, metric)

## Development

### Setup

```bash
git clone https://github.com/ctl-alt-leist/morphis-rs.git
cd morphis-rs
```

### Common Commands

| Command | Description |
|---------|-------------|
| `make lint` | Format and lint with rustfmt + clippy |
| `make test` | Run all tests |
| `make build` | Build release binary |
| `make clean` | Remove build artifacts |
| `make reset` | Clean and rebuild |

### Testing

Tests live in the `tests/` directory as integration tests:

```bash
make test                    # Run all tests
cargo test wedge             # Run tests matching "wedge"
```

### Code Style

- Rust 2024 edition
- `rustfmt` defaults, `clippy` clean
- Pre-commit hooks run fmt, clippy, and tests on every commit

## License

MIT License - see LICENSE file for details.

---

*Claude Code was used in the development of this project.*
