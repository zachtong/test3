# 3D Wafer Bonding -- Quick-Use Command Cheatsheet

One-page reference for every CLI script in this repo. Open this file
before running anything; do NOT ask the assistant for usage you can
look up here.

Conventions:
- `<sim>` = path to a converted 3D NPZ file (one simulation).
- `<folder>` = path to a folder of converted 3D NPZ files.
- `<tag>` = a training run's output tag, i.e. `outputs/<tag>/`.
- `<basis>` = a POD basis-cache file, i.e. `outputs/basis_cache/pod3d_*.npz`.
- All scripts accept `-h` / `--help` for the full flag list.
- Outputs default to per-script subpaths; override with `--out`.

---

## TL;DR -- run everything with one command

```bash
# Data-only (no training run yet): diversity + first 3 sims' single-sim viz
python scripts/viz_all.py --npz-dir <folder> --out viz/<short_name>

# Full suite (training run done): + POD viz + ML diagnostic + worst cases
python scripts/viz_all.py --npz-dir <folder> --tag <tag> \\
    --out viz/<tag>

# Pick the worst-error sims for the per-sim viz (needs --tag's results.json)
python scripts/viz_all.py --npz-dir <folder> --tag <tag> \\
    --select byerr --n-samples 5 --out viz/<tag>_worst5

# Skip the heavy interactive HTML, only generate static figures
python scripts/viz_all.py --npz-dir <folder> --tag <tag> \\
    --exclude interactive --out viz/<tag>
```

Re-runs are cheap (existing outputs skipped); pass `--force` to overwrite.
Outputs land under:

```
viz/<your_name>/
  diversity.png
  per_sim/<sim_basename>/
    topdown.gif          # 2D top-down (full disk, D2 mirror)
    kymo.png             # radial-slice trio at theta=0/45/90 (quarter)
    wafer_3d.gif         # 3D surface animation (full disk)
    wafer_3d_strip.png   # 3D snapshots at t=0/mid/final (full disk)
    interactive.html     # 3D Plotly (rotate / zoom / time slider)
  pod/{spectrum.png, mode_atlas.png}              # if --tag
  ml/{err_vs_floor.png, err_vs_floor_a6.png, ak_scatter.png}  # if --tag
  worst/0001_*.png ...                            # if --tag
```

Add the flat lower-wafer reference plane to ALL 3D viz in one shot:
```bash
python scripts/viz_all.py --npz-dir <folder> --out viz/<tag> --show-lower
```

See individual sections below for the manual single-script equivalents
(viz_all.py just shells out to them; copy any printed command to debug).

---

## Training

| Step | Command | Notes |
|---|---|---|
| Train a model end-to-end | `python scripts/train.py --config configs/default.yaml --data.npz_dir <folder> --data.workers 32 --pod.workers 32 --tag <tag>` | First run ~5-7 h (cache build + POD fit + train). Cache hits after that, ~30-60 min per repeat. `--data.workers None` is also safe now (auto-capped at 32; see `WAFER3D_LOADER_WORKERS_CAP`). |
| Try a different K | `... --pod.k 4 --tag <tag>_k4` | basis_cache HIT if `--pod.k <= k_cache (16)`; no refit. |
| Try bigger model | `... --model.channels 128 --tag <tag>_128ch` | basis_cache HIT, loader HIT. |
| Drop fewer leading steps | `... --data.drop_first_steps 0 --tag <tag>_nodrop` | Triggers fresh loader cache + basis refit (key changed). |

Tag rule: change `--tag` for every experiment so `outputs/<tag>/` is
distinct.

---

## NPZ inspection (no training needed)

| Goal | Command | Output |
|---|---|---|
| Scan whole folder for bad NPZs + run loader on 1-2 good ones | `python scripts/inspect_npz.py <folder> --n 2` | terminal report + PNG snapshots |
| Same, but skip slow Mode 1 schema scan | `python scripts/inspect_npz.py <folder> --skip-mode1 --n 2` | terminal + PNG, ~10-30 s |
| Quick check coords units / first-quadrant invariant | `python scripts/diagnose_coords.py <folder>` | terminal verdict |
| Plot displacement envelope over native time for 1 sim | `python scripts/diagnose_timeline.py <folder>` | terminal + PNG |
| Per-frame check of canonical field shape / nonzero counts | `python scripts/diagnose_loaded.py <folder>` | terminal stats |
| Scan tReal density (catches COMSOL adaptive-step skew) | `python scripts/diagnose_time_density.py <folder>` | terminal + PNG |
| Cross-sim summary table (units / dense window / contactTime) | `python scripts/diagnose_time_density.py <folder> --scan 20` | terminal table |

Default `--drop-first-steps` in diagnostics matches training (1).
Override with `--drop-first-steps 0` to inspect the raw pre-trim view.

---

## Visualisation

All viz lives under `scripts/viz_*.py` and shares `scripts/fieldviz/`
(WAFER_CMAP palette, D2 mirror helpers, bonded-mask computation,
provenance footer). Outputs go to whatever you pass `--out`, no
default directory. Recommended: `viz/<purpose>/<filename>` next to
your data folder.

### Single-sim viz (needs one NPZ)

| Goal | Command | Output |
|---|---|---|
| Top-down animation (canonical full disk, sensors + bonded contour) | `python scripts/viz_topdown_gif.py --sim <sim> --out viz/topdown.gif` | GIF ~500 KB |
| Override default and let each frame self-scale | `... --norm-mode per-frame` | default is per-sim so amplitude across frames is comparable; per-frame exaggerates within-frame structure but loses physical meaning |
| Add a raw-NPZ debug panel on the left (step_0000) | `... --include-raw` | (off by default; step_0000 is the pre-contact equilibration step, mostly near-zero) |
| 3D-kymograph trio (3 radial slices at theta=0/45/90) -- TALK HERO | `python scripts/viz_radial_kymograph.py --sim <sim> --out viz/kymo.png` | PNG ~100 KB |
| Interactive 3D surface (browser, rotate / zoom / hover / time-slider) | `python scripts/viz_interactive.py --sim <sim> --out viz/sim.html` | HTML ~10-25 MB |
| Same + flat lower-wafer reference plane (talks where gap matters) | `... --show-lower` | HTML, same size |
| 3D animated GIF of bonding process (embed in PPT / talk) | `python scripts/viz_3d_gif.py --sim <sim> --out viz/sim_3d.gif` | GIF ~1-3 MB |
| Same + lower wafer + custom camera (top-down isometric) | `... --show-lower --elev 80 --azim -90` | GIF |
| 3D static snapshot strip (3 panels: t=0/mid/final) -- PAPER FIG | `python scripts/viz_3d_strip.py --sim <sim> --out viz/sim_3d_strip.png` | PNG ~200 KB |
| Same + lower wafer reference | `... --show-lower` | PNG |

For viz_topdown_gif, default `--drop-first-steps 0` shows the raw
pre-contact step too (debugging-friendly); pass `--drop-first-steps 1`
to see only what training sees.

### Folder-level viz (needs a folder of NPZs)

| Goal | Command | Output |
|---|---|---|
| Cross-sim diversity / variance (std kymograph + top-down) | `python scripts/viz_diversity.py --npz-dir <folder> --out viz/diversity.png` | PNG ~150 KB |
| Same, but quick preview from 200 sims | `... --limit 200 --out viz/diversity_n200.png` | same; faster |
| Cap loader workers (fat node / shared box) | `... --workers 16` or `WAFER3D_LOADER_WORKERS_CAP=16 python ...` | safer on shared machines |
| Force diversity stats rebuild (folder changed) | `... --force` | rebuilds cache; ~50 MB |
| Skip diversity stats cache once (no overwrite) | `... --no-cache` | one-shot bypass |

Diversity stats cache: viz_diversity writes `<npz_dir>/_diversity_stats_<hash>.npz` (~50 MB) holding the (mean, var, n_eff) tensors. On any re-run with the same (folder, nx, ny, nt, drop, limit), the 93 GB loader read + Welford pass are SKIPPED and the figure renders in <1 s. Cache invalidates automatically when any key element changes. Pass `--no-cache` to bypass once, `--force` to rebuild AND overwrite. From `viz_all` use `--rebuild-diversity-cache` to plumb through.

Loader worker policy: when `--workers` is not set, the loader picks
`min(host_cores - 2, 32)` automatically. The 32 ceiling stops a 256-core
server from spawning ~254 processes (each with full-core BLAS) and
OOM'ing. Override the ceiling per-call with `--workers N` or globally
with the env var `WAFER3D_LOADER_WORKERS_CAP=N`. Cache HITS never spawn
workers regardless of this flag.

### POD-result viz (needs a basis file)

| Goal | Command | Output |
|---|---|---|
| sigma decay + cumulative energy (justify K choice) | `python scripts/viz_pod_spectrum.py --basis <basis> --K 8 --out viz/pod_spectrum.png` | PNG ~80 KB |
| 2x4 mode atlas (phi_k spatial shapes) -- POD HERO | `python scripts/viz_pod_mode_atlas.py --basis <basis> --K 8 --out viz/mode_atlas.png` | PNG ~250 KB |

To find a basis file: look in `outputs/basis_cache/pod3d_<hash>.npz`.
The hash depends on (data, grid, drop_first_steps, seed, fractions);
training prints it when fitting.

### ML diagnostic viz (needs results.json)

| Goal | Command | Output |
|---|---|---|
| Model error vs POD floor scatter (the headline) | `python scripts/viz_error_vs_floor.py --results outputs/<tag>/results.json --out viz/err_vs_floor.png` | PNG ~80 KB |
| Same, colour by a specific mode's per-sim error | `... --color-by a_6 --out viz/err_vs_floor_a6.png` | PNG, same |
| Per-mode error vs field error scatter (8 subplots) | `python scripts/viz_ak_scatter.py --results outputs/<tag>/results.json --out viz/ak_scatter.png` | PNG ~250 KB |

### Worst-case inspection (needs results.json + trained checkpoints)

| Goal | Command | Output |
|---|---|---|
| Render top-N worst test sims (GT vs pred vs error + a_k(t) curves) | `python scripts/viz_worst_cases.py --tag <tag> --topn 5 --out viz/worst/` | PNG per sim |
| Run on a different machine (override data path) | `... --data-dir-override /elsewhere/3d_npz` | same |
| Force worst-cases cache rebuild (rare) | `... --force` | rebuild + overwrite |
| Bypass cache once (no overwrite) | `... --no-cache` | one-shot |

First run: heavyweight. Loads dataset (93 GB), rebuilds split, loads
basis, loads 3 seed checkpoints, runs inference. ~1-2 min after
loader cache warm, 5-30 min cold. Then writes
`outputs/<tag>/_worst_top<N>_<ckpt_hash>.npz` (~100 MB) holding
(w_pred, w_true, a_pred, a_true, x_canon, y_canon, basenames, idx).

Re-runs: cache HIT reads the 100 MB and renders in seconds. Cache key
is (tag, topn, checkpoint fingerprint), so retraining auto-
invalidates and different topn values coexist as separate cache files
under the same tag directory. From `viz_all` use
`--rebuild-worst-cache` to force rebuild.

---

## Backwards compatibility / data prerequisites

- `viz_error_vs_floor.py`, `viz_ak_scatter.py`, `viz_worst_cases.py`
  need `results.json` with `per_sim_field_errs`,
  `per_sim_floor_errs`, `per_sim_basenames`, and (the last two)
  `per_sim_per_mode_errs`. These fields are written by every training
  run from the fieldviz-suite commit forward. An OLD `results.json`
  without them will print a clear error message.
- `viz_pod_spectrum.py` and `viz_pod_mode_atlas.py` need only the
  basis cache, which any training run writes to `cfg.basis_cache_dir`
  (default `outputs/basis_cache/`).

---

## Defaults baked in (so you do NOT have to remember)

- Quarter / full-disk:
  - Kymograph-style axes (radial slices) stay quarter (`viz_radial_kymograph`).
  - Top-down 2D heatmaps (`viz_topdown_gif`, `viz_diversity`,
    `viz_pod_mode_atlas`, `viz_worst_cases`, `viz_interactive`)
    mirror to full disk via D2.
- Colormap:
  - Signed displacement: TEL WAFER_CMAP (purple = deepest descent,
    yellow = rest / unbonded). Same as the 2D wafer_app GUI.
  - Unsigned (errors, std, mode-error scatter colour): viridis.
- Sensor markers: magenta (`SENSOR_PALETTE[4] = #DA1884`), never
  appears in WAFER_CMAP so stays visible at every value.
- Bonding-front overlay: TEL **orange** (`SENSOR_PALETTE[6] = #E16A13`).
  Was magenta originally; switched to orange because magenta sits too
  close to WAFER_CMAP's deep-purple end and visually confused the
  bonded region with the front itself.
- vmin/vmax policy:
  - Single-sim viz: per-sim global (stable across animation frames).
  - Multi-panel single-sim viz (e.g. radial kymograph trio): same
    per-sim range shared across the panels of one figure.
  - Cross-sim viz (diversity, error_vs_floor, ak_scatter, worst_cases):
    dataset-wide so sim-to-sim comparison is valid.
- Provenance footer: mandatory on EVERY viz. UTC timestamp + script +
  tag + sim id + hashes of input files.
- TEL palette anchor file: `scripts/fieldviz/palette.py` (1:1 port of
  the 2D `wafer_app/render/palette.py`).

---

## Tests

```bash
pytest tests/ -q
```

Currently 38+ tests across loader, POD basis, basis cache, fieldviz
helpers. All deterministic; no figures rendered (smoke for figures is
done manually via the commands above).
