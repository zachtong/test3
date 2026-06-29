# 3D Wafer Bonding NPZ Schema and Loader Contract

This document is the single source of truth for the on-disk format of one
converted simulation and for how the downstream loader consumes it. It
describes what the converter writes, what the loader reads, what the loader
deliberately ignores, and which project-level assumptions are baked into the
data path. When the converter or the loader changes in a way that breaks any
of the claims below, this file must be updated in the same commit.

## 1. Top-level invariants

- One NPZ corresponds to exactly one complete simulation. There is no
  many-simulation NPZ, and there is no NPZ that bundles a partial simulation
  (invalid or incomplete source JSON files are skipped, not repaired).
- A simulation is internally organized by COMSOL `waferData` step. Each step
  carries its own static mesh (coordinates), its own time-dependent fields,
  and its own scalar metadata. Cross-step the mesh is adaptive: the number of
  spatial points can change between steps because COMSOL refines the mesh
  during the bonding process.
- Within a step, the upper and lower wafer surfaces are split by z value: the
  lower wafer is the set of points where z equals the step's z_min, and the
  upper wafer is the set where z equals z_max. Counts on each side
  (N_lower_i and N_upper_i for step i) generally differ from each other and
  from step to step.
- Every time point inside every CONVERTED step is treated as one
  machine-learning sample. With T_i time points in converted step i, the
  total number of samples per NPZ is S = sum_i T_i (summed only over
  converted steps, see next bullet). A pair of sample-level index arrays
  maps every global sample index k back to the (step_idx, time_idx) pair
  that locates its data in the step-wise arrays.
- The converter ALWAYS drops the final original waferData step before
  conversion (the trailing step often holds a partial / restart frame that
  would distort the trajectory). Three keys make this explicit and
  recoverable:
    * `num_original_wafer_steps` is the source JSON's step count BEFORE
      removal.
    * `num_wafer_steps` is the post-removal converted count and is the
      ONLY authoritative step count for the loader; valid step indices
      run `[0, num_wafer_steps - 1]`.
    * `last_step_removed` is a bool scalar, always True under the current
      converter; if a future converter ever keeps the final step it must
      flip this to False.
  Loader code must treat `num_wafer_steps` -- not
  `num_original_wafer_steps` -- as the upper bound on `sample_step_index`
  and on the set of `step_{i:04d}_*` prefixes that exist on disk. This
  invariant also matches the 2D converter's "drop the trailing step"
  behavior (see `data/json_to_npz_converter.py` in
  `wafer_bonding_sparse_recon`).

## 2. Field inventory

The tables list every key the converter writes. Notation: `i` is a step
index formatted as four-digit zero-padded; `T_i`, `N_i`, `N_lower_i`,
`N_upper_i` are the per-step counts; `S` is the total sample count.

### 2.1 Step-wise arrays (prefix `step_{i:04d}_`)

| key suffix                          | shape                  | dtype     |
|-------------------------------------|------------------------|-----------|
| `coordinates_lower`                 | `(3, N_lower_i)`       | float32   |
| `coordinates_upper`                 | `(3, N_upper_i)`       | float32   |
| `displacement_z_corrected_lower`    | `(T_i, N_lower_i)`     | float32   |
| `displacement_z_corrected_upper`    | `(T_i, N_upper_i)`     | float32   |
| `thickness_lower`                   | `(T_i, N_lower_i)`     | float32   |
| `thickness_upper`                   | `(T_i, N_upper_i)`     | float32   |
| `bonding_front`                     | `(T_i,)`               | float32   |
| `tReal`                             | `(T_i,)`               | float64   |
| `num_time_points`                   | scalar                 | int64     |
| `num_points`                        | scalar                 | int64     |
| `num_lower_points`                  | scalar                 | int64     |
| `num_upper_points`                  | scalar                 | int64     |
| `z_min`                             | scalar                 | float64   |
| `z_max`                             | scalar                 | float64   |

Coordinate rows: row 0 is x, row 1 is y, row 2 is z. The converter
intentionally preserves z so a future loader can re-verify the upper/lower
split if needed; downstream training does not consume z. Coordinates are
static within a step and shared by all `T_i` time points of that step.

The corrected z displacement is computed during conversion using the COMSOL
shell variables: `displacement_z_corrected = shell.umz + 0.5 * shell.d *
arz`. This formula matches the 2D pipeline's correction and bakes the
mid-plane shell offset into the saved field. Note: even though the
correction is precomputed and the loader does NOT need `shell.d` to
reproduce it, `thickness_lower` / `thickness_upper` are still written to
the NPZ. They serve as provenance (auditing the correction) and leave a
hook open for any future model that wants the raw thickness as a separate
feature; the loader simply does not read them.

### 2.2 Global sample-index arrays (no step prefix)

| key                                       | shape | dtype     | purpose                                  |
|-------------------------------------------|-------|-----------|------------------------------------------|
| `sample_step_index`                       | `(S,)`| int64     | sample k -> step index `i`               |
| `sample_time_index_within_step`           | `(S,)`| int64     | sample k -> local time idx inside step i |
| `sample_tReal`                            | `(S,)`| float64   | convenience: duplicates step_{i}_tReal[time_idx] |
| `sample_bonding_front`                    | `(S,)`| float32   | convenience: duplicates step_{i}_bonding_front[time_idx] |
| `sample_num_time_points_in_step`          | `(S,)`| int64     | convenience                              |
| `sample_num_points`                       | `(S,)`| int64     | convenience                              |
| `sample_num_lower_points`                 | `(S,)`| int64     | convenience                              |
| `sample_num_upper_points`                 | `(S,)`| int64     | convenience                              |
| `sample_z_min`                            | `(S,)`| float64   | convenience                              |
| `sample_z_max`                            | `(S,)`| float64   | convenience                              |

The "convenience" arrays duplicate values already reachable through the
step-wise prefix; they exist so the loader can iterate samples linearly
without having to walk the step hierarchy.

### 2.3 File-level metadata (scalar arrays)

| key                         | dtype       | meaning                                                                                  |
|-----------------------------|-------------|------------------------------------------------------------------------------------------|
| `num_samples`               | int64       | S = total ML samples = `sum_i T_i` over the converted steps only                         |
| `num_wafer_steps`           | int64       | converted (post-removal) step count; the authoritative upper bound on `sample_step_index`|
| `num_original_wafer_steps`  | int64       | source JSON's step count BEFORE the converter dropped the trailing step                  |
| `num_valid_wafer_steps`     | int64       | converted steps that came through without internal failures                              |
| `last_step_removed`         | bool        | always True under the current converter; True iff `num_wafer_steps < num_original_wafer_steps` |
| `skipped_step_count`        | int64       | normally 0                                                                                |
| `step_metadata_json`        | string      | per-converted-step metadata serialized as JSON text (provenance / debugging)             |
| `skipped_steps_json`        | string      | per-skipped-step record serialized as JSON text (empty / `"[]"` when none were skipped)  |
| `source_json`               | string      | provenance: full source JSON path                                                         |
| `source_json_name`          | string      | provenance: source JSON basename                                                          |
| `json_file_size_bytes`      | int64       | provenance                                                                                |
| `converter_version`         | string      | converter version tag                                                                     |
| `minimal_fields`            | bool        | true if only the minimum field set was written                                            |
| `repaired_or_not`           | bool        | always False under the current "skip, do not repair" policy                               |
| `invalid_json_policy`       | string      | `"skip_no_repair"`                                                                        |
| `complete_json_required`    | bool        | True                                                                                      |
| `coordinate_system`         | string      | `"cartesian_3d"`                                                                          |
| `coordinate_layout`         | string      | `"(3,N)"`                                                                                 |
| `wafer_split_mode`          | string      | `"z_min_z_max"`                                                                           |
| `z_correction_mode`         | string      | `"shell_umz_plus_half_thickness_arz"`                                                     |
| `z_correction_formula`      | string      | `"displacement_z_corrected = shell.umz + 0.5 * shell.d * arz"`                            |
| `array_float_dtype`         | string      | `"float32"`                                                                               |
| `time_dtype`                | string      | `"float64"`                                                                               |

### 2.4 COMSOL physical metadata

Presence policy splits into two groups. The first group is ALWAYS written
to disk -- if the value is unavailable in the source JSON the converter
falls back to a NaN (for floats) or an empty string (for `modelName`), so
the key is present but the value flags "missing". The loader can rely on
the key existing and must defensively handle NaN.

| key                  | dtype                   | meaning                                                                   |
|----------------------|-------------------------|---------------------------------------------------------------------------|
| `contactTime`        | float64                 | physical time when initial contact is established; NaN if missing         |
| `releaseTime_LW`     | float64                 | physical time when the lower wafer is released; NaN if missing            |
| `releaseTime_UW`     | float64                 | physical time when the upper wafer is released; NaN if missing            |
| `hGap`               | float64                 | nominal initial gap between wafers; NaN if missing                        |
| `modelName`          | string                  | COMSOL model name; empty string if missing                                |

The second group is genuinely conditional and may be absent from a given
NPZ. The loader uses `_get_optional` so a missing key is treated as
"feature not provided".

| key                  | dtype                   | meaning                                                                   |
|----------------------|-------------------------|---------------------------------------------------------------------------|
| `allParams_json`     | string                  | full original `allParams` from COMSOL, serialized as JSON text; optional  |
| `expr`               | string array shape (7,) | expression list read from the FIRST successfully converted step; usually `["shell.umx","shell.umy","shell.umz","arx","ary","arz","shell.d"]` |

The minimal converted field set depends only on `shell.umz`, `arz`, and
`shell.d`. Extra expressions in `expr` may be present and are ignored.
`expr` is captured ONCE from the first converted step and not re-checked
against later steps -- if the source JSON were inconsistent (different
expression set across steps, which is not expected) only the first step's
list would be recorded here.

## 3. Loader access pattern

Read one sample k by:

```python
step_idx = int(data["sample_step_index"][k])
time_idx = int(data["sample_time_index_within_step"][k])
prefix = f"step_{step_idx:04d}"
coords_upper = data[f"{prefix}_coordinates_upper"]                      # (3, N_upper_i)
disp_upper   = data[f"{prefix}_displacement_z_corrected_upper"][time_idx]  # (N_upper_i,)
t_real       = data["sample_tReal"][k]                                  # scalar
b_front      = data["sample_bonding_front"][k]                          # scalar
```

Dtype-and-shape quick-reference for the index/scalar keys (important: only
true scalars are 0-d):

- `num_samples` and every file-level metadata scalar (e.g.
  `num_wafer_steps`, `last_step_removed`, `contactTime`) are 0-d arrays
  -- cast with `int(...)`, `float(...)`, or `bool(...)`.
- All `sample_*` arrays are 1-d shape `(S,)`. Index with `[k]` to get the
  k-th sample's value; the value itself is a 0-d numpy scalar.
- Per-step SCALAR metadata (`step_{i:04d}_num_time_points`,
  `step_{i:04d}_num_points`, `step_{i:04d}_num_lower_points`,
  `step_{i:04d}_num_upper_points`, `step_{i:04d}_z_min`,
  `step_{i:04d}_z_max`) are 0-d arrays.
- All OTHER `step_{i:04d}_*` keys -- coordinates, displacement,
  thickness, bonding_front, tReal -- are multi-dimensional arrays with
  the shapes given in section 2.1; they are NOT 0-d.

## 4. What the loader uses, what it ignores

This section pins down the project-level reading policy. The converter
stores more than the current training pipeline needs; the loader
deliberately reads a strict subset.

### 4.1 Used as training inputs

- `step_{i}_coordinates_upper[:2, :]` (x, y of upper wafer mesh).
- `step_{i}_displacement_z_corrected_upper[time_idx, :]`.
- `sample_step_index`, `sample_time_index_within_step`.
- `sample_tReal` for per-sample time tagging.
- `num_samples` as the per-NPZ sample count.

### 4.2 Used as `sim.params` metadata (diagnostic / future-feature, not part of `sim.f`)

- `sample_bonding_front` (kept for future label / anomaly use).
- `tReal` raw values, used to verify monotonicity and to drive time
  normalization.
- All file-level provenance (`source_json`, `converter_version`,
  `num_original_wafer_steps`, `num_wafer_steps`, `last_step_removed`,
  `step_metadata_json`, `skipped_steps_json`, etc.).
- All COMSOL physical metadata when present (`contactTime`,
  `releaseTime_LW`, `releaseTime_UW`, `hGap`, `modelName`,
  `allParams_json`, `expr`).
- Per-step counts (kept for sanity checks and debugging).

### 4.3 Read but discarded

- `step_{i}_coordinates_upper[2, :]` (z value of upper surface; only used
  by the converter to split surfaces by z, no use downstream).
- Convenience duplicates that the loader does not need: e.g.
  `sample_num_points`, `sample_num_lower_points`, `sample_z_min`,
  `sample_z_max`. The loader can fetch the same information through the
  step prefix if needed.

### 4.4 Not read at all

- `step_{i}_coordinates_lower`.
- `step_{i}_displacement_z_corrected_lower`.
- `step_{i}_thickness_lower`, `step_{i}_thickness_upper`.

Reasons for excluding the lower wafer and the thickness fields:

- The simulation setup constrains the lower wafer to its prescribed pre-bond
  shape (it deforms only in the initial transient and is effectively static
  thereafter), matching the 2D pipeline's convention. The sensor side of
  the experiment also observes only the upper wafer. The mid-plane
  correction that benefits from thickness is already baked into
  `displacement_z_corrected_upper` at conversion time.

If a future project phase needs the lower wafer or raw thickness (for
example, an extension that models substrate compliance, or an anomaly
detector that inspects the lower wafer's shape), it does not require a
schema change: those fields are still on disk and can simply start being
read.

## 5. Project assumptions baked into the loader

These assumptions hold for the current data and the current pipeline.
Changing any of them requires editing both the loader and this document.

### 5.1 Quarter-disk symmetry

The simulation enforces 90-degree rotational symmetry about the z axis, so
the converter only writes points in the first quadrant (x >= 0, y >= 0,
x^2 + y^2 <= R^2 in physical units; normalized to x^2 + y^2 <= 1).
Consequences for the loader and downstream code:

- The canonical grid covers `[0, 1] x [0, 1]` in normalized x and y, not
  `[-1, 1] x [-1, 1]`.
- The disk mask is the quarter-disk mask `x >= 0 AND y >= 0 AND
  x^2 + y^2 <= 1`. Off-disk cells are filled with zero before POD.
- Sensor positions in polar `(r, theta_deg)` are constrained to
  `theta in [0, 90]`. A sensor outside this range is a configuration
  error.
- For visualization only, the quarter field can be mirrored across the x
  and y axes to display the full disk. Mirroring never happens inside the
  training data path.

### 5.2 Upper wafer is the field of interest

`sim.f` has shape `(Nx, Ny, Nt)` and equals the upper wafer's mid-plane
corrected z displacement on the canonical grid. There is no subtraction
between upper and lower, and no addition of an initial-gap offset. The
training signal is the relative deformation history of the upper wafer
alone.

### 5.3 POD only

No co-moving shift, no sPOD, no front-channel target. The basis fitted in
`core/pod_basis.py` is the lab-frame POD of `sim.f`. `bonding_front` is
read but is NOT part of `sim.f` and NOT predicted by the model in this
phase.

## 6. Validation rules the loader enforces

When reading one NPZ, the loader must:

- Treat `num_samples` as the authoritative S; cross-check that
  `sample_step_index.shape == (S,)` and that the maximum step index is
  `< num_wafer_steps` (the converted count, NOT
  `num_original_wafer_steps`).
- For each step encountered: read `coordinates_upper`, verify
  `coords.shape[1] == num_upper_points`, and that all points satisfy
  `x >= 0` and `y >= 0` within a small tolerance. A negative coordinate
  is a converter or schema error.
- Build a single Delaunay triangulation (via SciPy
  `LinearNDInterpolator`) per step and reuse it for every time index in
  that step.
- After concatenating samples across steps, normalize the global time
  axis via `s = (tReal - tReal.min()) / (tReal.max() - tReal.min())`.
  At step boundaries `s` can have duplicate or sub-step-overlap values;
  follow the 2D pattern: `np.unique` collapses exact duplicates, and a
  backward jump larger than the median forward `dt` aborts the load
  with a clear error.
- Interpolate the canonicalized (Nx, Ny) frames onto the canonical time
  grid `t_canon = np.linspace(0, 1, Nt)` using row-wise linear
  interpolation. Off-disk cells stay zero throughout.

## 7. Loader pipeline summary

The intended one-pass pipeline per NPZ:

1. Open the NPZ. Read `num_samples`, the two sample-level index arrays,
   `sample_tReal`, `sample_bonding_front`, and file / COMSOL metadata.
2. Group sample indices by `sample_step_index`.
3. For each unique step:
   a. Read `coordinates_upper[:2, :].T` once: an `(N_upper_i, 2)` array
      of native (x, y) points.
   b. Build one `LinearNDInterpolator` over those points (the Delaunay
      cost is paid once per step, not per sample).
   c. For each sample in that step: index the time slice
      `displacement_z_corrected_upper[time_idx, :]`, feed it to the
      interpolator at the canonical `(Nx, Ny)` grid query points, and
      store the resulting `(Nx, Ny)` slice in the per-sim stack
      `f_stack` at the right sample position.
4. Apply the quarter-disk mask: cells with `x^2 + y^2 > 1` are set to 0
   (NaNs from the interpolator outside the convex hull are also zeroed).
5. Resample along the time axis from native `sample_tReal` (normalized to
   `[0, 1]`) onto the canonical `t_canon`, producing the final
   `f (Nx, Ny, Nt)`.
6. Return a `Simulation(f=f, params={...})` with all metadata in
   `params`.

The aggregated cache across all sims is written as a single uncompressed
NPZ in the dataset folder, reusing the cache framework that the 2D pipeline
already validated (parallel build, atomic write to `.tmp` then rename,
stale-`.tmp` sweep on next load, ENOSPC-resilient cache write).

## 8. Open items / future work

- A converter-side check that confirms x and y of every coordinate array
  satisfy the quarter-disk invariant before writing. The loader will
  enforce it on read, but pre-flight at write time is cheaper.
- A diagnostic script that, for a small set of converted NPZs, plots the
  first and last sample's upper field overlaid with sensor positions, to
  catch unit or orientation errors before a full cache build.
- A randomized SVD path for `core/pod_basis.py` so spatial resolutions
  above Nx = Ny = 128 stay tractable. Not needed at the current target
  resolution.
