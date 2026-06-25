# Wafer Bonding Sparse Reconstruction -- 3D

Reconstruct the full spatiotemporal gap field `f(x, y, t)` of a wafer-bonding
process from a sparse set of fixed-position sensor time series. Unlike the
axisymmetric 2D variant (`wafer_bonding_sparse_recon`), this codebase makes
NO axisymmetry assumption: the field lives on a 2D Cartesian grid
`(Nx, Ny, Nt)` so non-symmetric bonding fronts and defects are first-class.

## Method (one paragraph)

Lab-frame POD (no co-moving shift) over the flattened spatial axis
`Nx * Ny -> K` modes; a BiTCN predicts the K POD coefficients per timestep
from the n sensor traces. Sensors are placed on a polar `(r, theta)` schedule
mapped to nearest Cartesian grid indices, so the rig-side intuition of "an
edge sensor on the X axis" is preserved.

## Status

Early scaffolding. Generic infrastructure (BiTCN, training loop, normalization,
loss, metrics, JSON result container) is ported from the 2D codebase. 3D-specific
domain code (`core/simulation.py`, `core/grid.py`, `core/sensors.py`,
`core/pod_basis.py`) is written. The NPZ schema and the `data/loader.py` /
`data/json_to_npz_converter.py` are STUB -- they raise `NotImplementedError`
until the 3D simulation export format is finalized.

## Setup

```bash
# install PyTorch for your platform/CUDA first
pip install torch
pip install -r requirements.txt
```

## Structure

- `core/` -- 3D Simulation, Cartesian grid, polar sensor placement, POD basis
- `models/` -- BiTCN (verbatim from 2D; spatial-dim-agnostic)
- `training/` -- training loop, config, loss (verbatim or POD-only adapted)
- `evaluation/` -- metrics, ResultSet, scorer, baselines + 2D fieldviz
- `data/` -- loader + JSON->NPZ converter (both STUB pending schema)
- `scripts/` -- CLI entry points
- `configs/` -- experiment YAMLs
- `tests/` -- fixture tests (to be written)

## Relationship to the 2D codebase

This is a hard fork of `wafer_bonding_sparse_recon` at the point the 3D effort
started. Files duplicated here are independent copies -- fixes do not auto-
propagate. If a generic-infra fix lands on either side, mirror it manually.
