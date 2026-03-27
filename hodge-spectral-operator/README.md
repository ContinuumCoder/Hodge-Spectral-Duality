# Hodge Spectral Operator (HSO)

Neural operator learning on manifolds with Hodge decomposition inductive bias.

## Features

- **Any geometric input**: mesh, point cloud, or graph
- **Any differential form task**: 0-form→0-form, 0-form→1-form, etc.
- **Hodge inductive bias**: de Rham cross-form operators as physics priors
- **Spectral bilinear layers**: pseudo-spectral quadratic nonlinearity
- **Adaptive dual-branch**: neural gating between spectral and spatial
- **One-line API**: `model.fit(X, Y)` / `model.predict(X_new)`

## Quick Start

```python
from hodge_spectral import HodgeOperator

# From mesh
model = HodgeOperator.from_mesh(points, faces, task="0to1", k=64)
model.fit(X_train, Y_train, epochs=100)
Y_pred = model.predict(X_test)
metrics = model.evaluate(X_test, Y_test)

# From point cloud (auto-triangulates)
model = HodgeOperator.from_pointcloud(points, task="0to0")

# From graph
model = HodgeOperator.from_graph(edge_index, n_nodes, positions, task="0to1")
```

## Install

```bash
pip install hodge-spectral-operator
```

## Architecture

```
Input → Spectral Lift (Φ₀, Φ₁, Φ₂)
  ↓
De Rham cross-terms: div(c₁), grad(c₀)
  ↓
SpectralBilinearLayer: linear + c₀⊙δ(c₁), c₁⊙d(c₀)
  ↓
┌─ Spectral Branch → Φ₀ coefficients → decode ─┐
│                                                │
├─ Spatial Branch → direct N-dim residual ───────┤
│                                                │
├─ Neural Gate → adaptive α(x) mixing ──────────┤
│                                                │
└─ Commutator → base/res disagreement fix ──────┘
  ↓
Output prediction
```

## Benchmarks

| Task | Form | HSD | FNO | DeepONet |
|------|------|-----|-----|---------|
| Ellipsoid Aero | 0→1 | **0.025** | 0.259 | 0.113 |
| Torus Helmholtz | 0→1 | **0.049** | 0.418 | 0.277 |

## Built-in Examples

```bash
# Ellipsoid external aerodynamics (genus-0, 0-form → 1-form)
python examples/example_ellipsoid_aero.py

# Torus Helmholtz vortex flow (genus-1, 0-form → 1-form)
python examples/example_torus_helmholtz.py
```
