# Morphis

A Rust implementation of geometric algebra, ported from the Python [morphis](https://github.com/ctl-alt-leist/morphis) library. The name derives from Greek *morphe* (form) — embodying the transformation and adaptation of geometric structures across different contexts while preserving their essential nature.

## Features

- **Geometric Algebra Core**: Vectors (k-vectors), multivectors, and operations (wedge, geometric product, interior products)
- **Metric-Aware**: Objects carry their metric context (Euclidean, Lorentzian, Projective)
- **Full Tensor Storage**: Antisymmetric tensor representation enabling direct contraction operations

## Documentation

- [Project Overview](docs/0_project-overview.md) — Vision and scope
- [Concepts](docs/1_concepts/) — Mathematical foundations (vectors, products, duality, metric)

## Quick Start

```rust
use morphis::metric::euclidean;
use morphis::vector::basis;
use morphis::ops::{wedge, geometric};

// Create a 3D Euclidean metric and basis vectors
let g = euclidean::<3>();
let e = basis(g);

// Wedge product: oriented plane
let b = wedge(&e[0], &e[1]);

// Geometric product: scalar + bivector
let m = geometric(&e[0], &e[1]);
```

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

Tests are co-located with source in `#[cfg(test)] mod tests` blocks:

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
