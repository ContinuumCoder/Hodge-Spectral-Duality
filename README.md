
This repository provides code for learning divergence-free vector fields on manifolds using spectral methods based on the Hodge decomposition and discrete exterior calculus.

---

## Overview

We propose **HSD (Hodge-Spectral Duality)**, a physics-informed neural operator that learns mappings between differential forms on manifolds. Our approach leverages the deep connection between algebraic topology and spectral geometry through:

- **Discrete Exterior Calculus (DEC)** for intrinsic geometric representations
- **Hodge Decomposition Theorem** for physically meaningful field decomposition
- **Spectral bases of Hodge Laplacians** across all form degrees (0, 1, 2)
- **De Rham complex operators** for topologically consistent transformations

The key innovation is a **dual-branch architecture** that separates:
1. **Topological Base (Spectral Branch)**: Captures global, topologically invariant features in the spectral domain
2. **Geometric Fiber (Local Branch)**: Learns fine-grained spatial variations in the physical domain

---

## Theoretical Foundation

### Differential Forms and the de Rham Complex

On a 2-dimensional manifold embedded in R^3, we work with the de Rham complex:

```
Ω^0(M) --d0--> Ω^1(M) --d1--> Ω^2(M)
   ↑              ↑              ↑
0-forms       1-forms        2-forms
(scalars)    (vectors)      (densities)
```

where:
- **0-forms**: Scalar fields on vertices (e.g., temperature, pressure)
- **1-forms**: Vector fields along edges (e.g., velocity, flux)
- **2-forms**: Density fields on faces (e.g., vorticity, divergence)

The exterior derivative `d` and codifferential `δ` satisfy:
- `d ∘ d = 0` (closed forms are locally exact)
- `δ = ±*d*` (Hodge dual relationship)

### Hodge Decomposition Theorem

Any differential form ω on a compact Riemannian manifold admits a unique orthogonal decomposition:

```
ω = dα + δβ + h
```

where:
- `dα` is the **exact component** (gradient of a potential)
- `δβ` is the **coexact component** (curl of a vector potential)
- `h` is the **harmonic component** (satisfies Δh = 0)

The harmonic forms encode **topological invariants**: their dimension equals the Betti numbers of the manifold. This decomposition is central to our approach—we learn operators that respect this structure.

### Hodge Laplacians and Spectral Bases

For each form degree k, the Hodge Laplacian is defined as:

```
Δ_k = dδ + δd
```

In the discrete setting (DEC):
- `Δ_0 = B_1^T B_1` (graph Laplacian on nodes)
- `Δ_1 = B_1 B_1^T + B_2^T B_2` (edge Laplacian)
- `Δ_2 = B_2 B_2^T` (face Laplacian)

where `B_1` and `B_2` are the signed incidence matrices (discrete boundary operators).

The eigenfunctions of these Laplacians form orthonormal bases:
- `Φ_0 = {φ_0^1, φ_0^2, ...}` for 0-forms (node functions)
- `Φ_1 = {φ_1^1, φ_1^2, ...}` for 1-forms (edge functions)
- `Φ_2 = {φ_2^1, φ_2^2, ...}` for 2-forms (face functions)

### Spectral de Rham Operators

The key insight is that the de Rham operators can be represented in the spectral domain:

```
M_{d0} = Φ_1^T B_1 Φ_0    (Spectral Gradient)
M_{δ1} = Φ_0^T B_1^T Φ_1  (Spectral Divergence)
M_{d1} = Φ_2^T B_2 Φ_1    (Spectral Curl)
M_{δ2} = Φ_1^T B_2^T Φ_2  (Spectral Co-curl)
```

These matrices encode how differential operators transform spectral coefficients, enabling physics-aware learning in a compressed representation.

### Intrinsic Lifting: From Scalars to Forms

Given a scalar input field f (0-form), we derive the full de Rham representation through intrinsic lifting:

```
f ∈ Ω^0  →  c_0 = Φ_0^T f           (0-form coefficients)
         →  c_1 = Φ_1^T (B_1 f)     (1-form: gradient)
         →  c_2 = Φ_2^T (B_2 B_1 f) (2-form: curl of gradient = 0)
```

The constraint `d^2 = 0` manifests as `c_2 ≈ 0` for lifted signals, providing a consistency check.

---

## HSD Architecture: Dual-Branch System

### Conceptual Framework

The HSD model implements a **base-fiber decomposition** inspired by fiber bundles in differential geometry:

```
                    ┌─────────────────────────────┐
                    │     Input: (c_0, c_1, c_2)  │
                    │   Spectral Coefficients     │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼                                         ▼
    ┌─────────────────────┐               ┌─────────────────────┐
    │   TOPOLOGICAL BASE  │               │   GEOMETRIC FIBER   │
    │   (Spectral Branch) │               │   (Local Branch)    │
    ├─────────────────────┤               ├─────────────────────┤
    │ • Physics Encoder   │               │ • FNO on 3D Grid    │
    │ • De Rham Coupling  │               │ • Position MLP      │
    │ • Spectral MLP      │               │ • Local Refinement  │
    │                     │               │                     │
    │ Output: Base field  │               │ Output: Residual    │
    │ (global, smooth)    │               │ (local, detailed)   │
    └──────────┬──────────┘               └──────────┬──────────┘
              │                                         │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  Orthogonal Decomposition   │
                    └─────────────────────────────┘
```

### Topological Base Branch

The spectral branch operates entirely in the Hodge-spectral domain:

1. **Physics Encoder**: Computes cross-form interactions using de Rham operators
   ```
   feat_0 = [c_0, M_{δ1} c_1]           # 0-form + divergence of 1-form
   feat_1 = [c_1, M_{d0} c_0, M_{δ2} c_2]  # 1-form + gradient + co-curl
   feat_2 = [c_2, M_{d1} c_1]           # 2-form + curl of 1-form
   ```

2. **Spectral Amplification**: Frequency-dependent gain to balance mode contributions

3. **Physics-Aware gMLP**: Gated MLP layers that preserve spectral structure

4. **Output**: Spectral coefficients for the base vector field, transformed back via `Φ_0`

### Geometric Fiber Branch

The local branch captures fine-grained spatial variations:

1. **FNO Sub-network**: 3D Fourier Neural Operator on a regular grid embedding
2. **Coupling MLP**: Position-aware network conditioned on spectral features
3. **Output**: Residual vector field in physical space

---

## Repository Structure

```
Hodge-Spectral-Duality/
│
├── externalAerodynamics/
│   ├── flux_field_data/
│   │   └── (generated data files)
│   ├── config.py
│   ├── dataset.py
│   ├── generate_data.py
│   ├── main.py
│   ├── models.py
│   ├── spectral_operators.py
│   ├── topo_metrics.py
│   ├── training.py
│   ├── utils.py
│   └── visualization.py
│
├── magnetostatics/
│   ├── flux_field_data/
│   │   └── (generated data files)
│   ├── config.py
│   ├── dataset.py
│   ├── generate_data.py
│   ├── main.py
│   ├── models.py
│   ├── spectral_operators.py
│   ├── topo_metrics.py
│   ├── training.py
│   ├── training copy.py
│   ├── utils.py
│   └── visualization.py
│
├── toroidalTransport/
│   ├── config.py
│   ├── dataset.py
│   ├── generate_data.py
│   ├── main.py
│   ├── models.py
│   ├── spectral_operators.py
│   └── topo_metrics.py
│
└── README.md
```

---

## Installation

### Requirements

```bash
# Create conda environment
conda create -n hsd python=3.10
conda activate hsd

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn matplotlib

# Geometry and topology
pip install pyvista toponetx

# Neural operators (optional, for baselines)
pip install torch-geometric
pip install neuraloperator

# Visualization
pip install pillow pandas
```

### Verify Installation

```python
import torch
import toponetx
import pyvista
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Quick Start

### 1. Generate Data

Each task requires generating synthetic physics data:

```bash
cd externalAerodynamics
python generate_data.py
```

This creates:
- `flux_field_data/flux_field_dataset.pkl` — Training/test data
- `flux_field_data/preview_samples.png` — Visualization of samples

### 2. Train Models

```bash
python main.py
```

On first run, you will be prompted:
```
EXISTING TRAINED MODELS DETECTED
Options:
  [1] Load existing models and skip training
  [2] Retrain all models from scratch
  [3] Train only missing models
```

### 3. View Results

Results are saved to `output_flux/`:
- `training_log.txt` — Full experiment log
- `training_curves.png` — Loss curves for all models
- `metrics_comparison.png` — Bar chart of topology metrics
- `flux_samples/` — Per-sample visualizations
- `topo_metrics.json` — Quantitative results

---

## Tasks

### Task 1: External Aerodynamics

**Geometry**: Ellipsoid surface (configurable axes)  
**Input**: Vorticity field (scalar, 0-form)  
**Output**: Velocity field (tangent vector, 1-form representation)  
**Physics**: Poisson stream function with global moment coupling

The velocity field is computed as:
```
u = n × ∇ψ + Σ c_i · basis_i
```
where ψ solves the Laplace-Beltrami Poisson equation and the global flow direction is determined by vorticity moments.

```bash
cd externalAerodynamics && python main.py
```

### Task 2: Magnetostatics

**Geometry**: Complex 3D surfaces with varying curvature  
**Input**: Current density distribution  
**Output**: Magnetic flux density (tangent field)  
**Physics**: Biot-Savart law discretized on manifolds

```bash
cd magnetostatics && python main.py
```

### Task 3: Toroidal Transport

**Geometry**: Torus and toroidal-like manifolds  
**Input**: Source/sink distribution  
**Output**: Transport flux field  
**Physics**: Steady-state advection-diffusion on curved surfaces

This task specifically tests the method's ability to handle non-trivial topology (genus-1 surfaces with non-zero first Betti number).

```bash
cd toroidalTransport && python main.py
```

---

## Evaluation Metrics

### Physics Metrics

| Metric | Description |
|--------|-------------|
| MSE | Mean squared error on vector field |
| Divergence Fidelity | Conservation of mass (div u = 0) |
| Curl MSE | Vorticity reconstruction accuracy |
| Vorticity Fidelity | Correlation of vorticity fields |
| Enstrophy Fidelity | Total squared vorticity preservation |
| Energy Fidelity | Total kinetic energy matching |

### Topology Metrics

Computed on the **vorticity field** (not velocity magnitude) for stability:

| Metric | Description |
|--------|-------------|
| Betti-0 Score | Connected component counting at multiple thresholds |
| Level Set IoU | Overlap of vorticity iso-surfaces |
| Vortex Count Accuracy | Number of vortex cores (local maxima of vorticity) |

**Why vorticity?** For vortex-dominated flows, velocity magnitude vanishes at vortex centers (saddle points), creating unstable "donut" topologies. Vorticity peaks at vortex centers, giving stable connected regions.

### Spectral Metrics

| Metric | Description |
|--------|-------------|
| Gradient Fidelity | Directional derivative correlation |
| Spectral Fidelity | Eigenmode-weighted coefficient matching |

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [TopoNetX](https://github.com/pyt-team/TopoNetX) for simplicial complex operations
- [PyVista](https://pyvista.org/) for 3D visualization
- [NeuralOperator](https://github.com/neuraloperator/neuraloperator) for FNO implementation
