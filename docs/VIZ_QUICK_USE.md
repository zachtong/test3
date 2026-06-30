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

## Training

| Step | Command | Notes |
|---|---|---|
| Train a model end-to-end | `python scripts/train.py --config configs/default.yaml --data.npz_dir <folder> --data.workers 32 --pod.workers 32 --tag <tag>` | First run ~5-7 h (cache build + POD fit + train). Cache hits after that, ~30-60 min per repeat. |
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
| Top-down 2-panel animation (raw vs canonical, sensors + bonded contour) | `python scripts/viz_topdown_gif.py --sim <sim> --out viz/topdown.gif` | GIF ~500 KB |
| 3D-kymograph trio (3 radial slices at theta=0/45/90) -- TALK HERO | `python scripts/viz_radial_kymograph.py --sim <sim> --out viz/kymo.png` | PNG ~100 KB |
| Interactive 3D surface (browser, rotate / zoom / hover / time-slider) | `python scripts/viz_interactive.py --sim <sim> --out viz/sim.html` | HTML ~10-25 MB |

For viz_topdown_gif, default `--drop-first-steps 0` shows the raw
pre-contact step too (debugging-friendly); pass `--drop-first-steps 1`
to see only what training sees.

### Folder-level viz (needs a folder of NPZs)

| Goal | Command | Output |
|---|---|---|
| Cross-sim diversity / variance (std kymograph + top-down) | `python scripts/viz_diversity.py --npz-dir <folder> --out viz/diversity.png` | PNG ~150 KB |
| Same, but quick preview from 200 sims | `... --limit 200 --out viz/diversity_n200.png` | same; faster |

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

Heavyweight: loads dataset, rebuilds split, fits or loads basis, loads
3 seed checkpoints, runs inference. ~1-2 min after caches warm.

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
- Bonding-front overlay: TEL magenta line / contour.
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
