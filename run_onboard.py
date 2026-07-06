"""Onboard a new NPZ dataset through 6 diagnostic layers before
committing training compute. Never trust a new dataset -- catch
converter bugs, physics anomalies, and pipeline mismatches BEFORE
the 93 GB F build + POD fit + training.

Layer sequence (each aborts if the previous fails):
    0. Basic sanity      -- inspect_npz.py       (file count, schema)
    1. GT quality        -- inspect_gt_quality.py (tReal + rim kink)
    2. Data diversity    -- viz_diversity.py + diagnose_time_density
    3. Loader smoke      -- diagnose_loaded.py    (small subset)
    4. POD completeness  -- probe train + spectrum + mode atlas
    5. Trial train       -- ~1000 sims, 50 epochs, single seed

Every layer writes into a per-dataset report folder:
    viz/onboard_<tag>/
        layer0_inspect_npz.txt
        layer1_gt_quality/                  # inspect_gt_quality output
        layer1_gt_summary.txt
        layer2_diversity.png
        layer2_time_density.png
        layer3_loaded/                      # diagnose_loaded output
        layer4_probe/                       # POD probe run outputs
        layer4_spectrum.png
        layer4_mode_atlas.png
        layer5_trial/                       # trial train outputs
        REPORT.md                           # overview + verdict

Run:
    python run_onboard.py --npz-dir /path/to/new_dataset --tag mynew

If any layer fails or reports issues, subsequent layers are skipped
and REPORT.md flags which layer stopped the run.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable


def _step(name: str, cmd: list[str], log_path: Path,
           cwd: Path | None = None) -> tuple[bool, float, str]:
    """Run a subprocess, tee stdout+stderr to log_path, return
    (ok, duration_sec, tail)."""
    print(f"\n===== {name} =====", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fp:
        fp.write(f"# {name}\n# {' '.join(cmd)}\n\n")
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True, cwd=cwd)
        except FileNotFoundError as e:
            fp.write(f"[MISSING] {e}\n")
            print(f"  MISSING: {e}", flush=True)
            return False, time.time() - t0, str(e)
        fp.write(proc.stdout)
    ok = proc.returncode == 0
    tail = "\n".join(proc.stdout.splitlines()[-8:])
    print(tail, flush=True)
    print(f"  -> {'OK' if ok else 'FAIL'} in "
          f"{time.time() - t0:.1f}s "
          f"(log: {log_path})", flush=True)
    return ok, time.time() - t0, tail


def layer0_inspect_npz(npz_dir: Path, out_dir: Path
                        ) -> tuple[bool, str]:
    ok, dt, tail = _step(
        "Layer 0: inspect_npz",
        [PY, "scripts/inspect_npz.py", str(npz_dir)],
        out_dir / "layer0_inspect_npz.txt")
    return ok, tail


def layer1_gt_quality(npz_dir: Path, out_dir: Path,
                       n_sample: int) -> tuple[bool, str]:
    layer_dir = out_dir / "layer1_gt_quality"
    ok, dt, tail = _step(
        "Layer 1: inspect_gt_quality",
        [PY, "scripts/inspect_gt_quality.py",
         "--npz-dir", str(npz_dir),
         "--n-check", str(n_sample),
         "--out", str(layer_dir)],
        out_dir / "layer1_gt_quality.txt")
    if not ok:
        return ok, tail
    summary_json = layer_dir / "summary.json"
    if summary_json.is_file():
        _step(
            "Layer 1b: dump_gt_summary",
            [PY, "scripts/dump_gt_summary.py",
             str(summary_json)],
            out_dir / "layer1_gt_summary.txt")
    return True, tail


def layer2_diversity(npz_dir: Path, out_dir: Path,
                      nx: int, ny: int, nt: int,
                      showcase_n: int) -> tuple[bool, str]:
    ok1, _, tail1 = _step(
        "Layer 2a: viz_diversity",
        [PY, "scripts/viz_diversity.py",
         "--npz-dir", str(npz_dir),
         "--nx", str(nx), "--ny", str(ny), "--nt", str(nt),
         "--out", str(out_dir / "layer2_diversity.png")],
        out_dir / "layer2_diversity.txt")
    ok2, _, tail2 = _step(
        "Layer 2b: diagnose_time_density",
        [PY, "scripts/diagnose_time_density.py",
         str(npz_dir),
         "--out", str(out_dir / "layer2_time_density.png")],
        out_dir / "layer2_time_density.txt")
    ok3, tail3 = _layer2c_sim_showcase(
        npz_dir, out_dir, nx, ny, nt, showcase_n)
    return (ok1 and ok2 and ok3),\
        f"{tail1}\n---\n{tail2}\n---\n{tail3}"


def _layer2c_sim_showcase(npz_dir: Path, out_dir: Path,
                            nx: int, ny: int, nt: int,
                            n: int) -> tuple[bool, str]:
    """Sample n sims (by sorted filename, deterministic) and render
    top-down GIF + 5-angle radial kymograph + 3D strip for each so
    the user can eyeball what typical sims in the dataset look like.
    All viz here works on a SINGLE raw NPZ; no trained model needed."""
    showcase_dir = out_dir / "layer2_showcase"
    showcase_dir.mkdir(parents=True, exist_ok=True)
    sims = sorted(p for p in npz_dir.iterdir()
                    if p.suffix == ".npz" and
                    not p.name.startswith("_loader_cache_"))
    if not sims:
        return False, "no .npz files found for showcase"
    picks = sims[:n]
    all_ok = True
    tails: list[str] = []
    for sim in picks:
        stem = sim.stem
        # top-down 2D GIF (bonding front propagation over time)
        ok_td, _, tail_td = _step(
            f"Layer 2c/{stem}: viz_topdown_gif",
            [PY, "scripts/viz_topdown_gif.py",
             "--sim", str(sim),
             "--out", str(showcase_dir / f"{stem}_topdown.gif"),
             "--nx", str(nx), "--ny", str(ny), "--nt", str(nt)],
            showcase_dir / f"{stem}_topdown.txt")
        # 5-angle radial kymograph (r-t heatmap per angle)
        ok_ky, _, tail_ky = _step(
            f"Layer 2c/{stem}: viz_radial_kymograph",
            [PY, "scripts/viz_radial_kymograph.py",
             "--sim", str(sim),
             "--out", str(showcase_dir / f"{stem}_kymo.png"),
             "--nx", str(nx), "--ny", str(ny), "--nt", str(nt)],
            showcase_dir / f"{stem}_kymo.txt")
        # 3D snapshot strip (multiple frames side by side)
        ok_st, _, tail_st = _step(
            f"Layer 2c/{stem}: viz_3d_strip",
            [PY, "scripts/viz_3d_strip.py",
             "--sim", str(sim),
             "--out", str(showcase_dir / f"{stem}_3dstrip.png"),
             "--nx", str(nx), "--ny", str(ny), "--nt", str(nt),
             "--show-lower"],
            showcase_dir / f"{stem}_3dstrip.txt")
        sim_ok = ok_td and ok_ky and ok_st
        all_ok = all_ok and sim_ok
        tails.append(f"{stem}: td={ok_td} ky={ok_ky} strip={ok_st}")
    return all_ok, "\n".join(tails)


def layer3_loader_smoke(npz_dir: Path, out_dir: Path,
                          nx: int, ny: int, nt: int
                          ) -> tuple[bool, str]:
    ok, _, tail = _step(
        "Layer 3: diagnose_loaded",
        [PY, "scripts/diagnose_loaded.py",
         str(npz_dir),
         "--nx", str(nx), "--ny", str(ny), "--nt", str(nt)],
        out_dir / "layer3_loaded.txt")
    return ok, tail


def layer4_pod_probe(npz_dir: Path, out_dir: Path, tag: str,
                       nx: int, ny: int, nt: int,
                       probe_limit: int, k_cache: int
                       ) -> tuple[bool, str]:
    probe_tag = f"onboard_{tag}_probe"
    ok_train, _, tail_train = _step(
        "Layer 4a: POD probe train (K=32, few epochs, few sims)",
        [PY, "scripts/train.py",
         "--config", "configs/default.yaml",
         "--data.npz_dir", str(npz_dir),
         "--data.nx", str(nx), "--data.ny", str(ny),
         "--data.nt", str(nt),
         "--data.limit", str(probe_limit),
         "--pod.k", str(k_cache),
         "--train.epochs", "5",
         "--seeds", "[7]",
         "--tag", probe_tag],
        out_dir / "layer4_train.txt")
    if not ok_train:
        return False, tail_train

    # Find the basis cache the probe just wrote (or reused).
    basis_dir = Path("outputs/basis_cache")
    if basis_dir.is_dir():
        candidates = sorted(basis_dir.glob("pod3d_*.npz"),
                             key=lambda p: p.stat().st_mtime,
                             reverse=True)
        basis_file = candidates[0] if candidates else None
    else:
        basis_file = None
    if basis_file is None:
        return False, "no basis_cache file found after probe"

    ok_spec, _, tail_spec = _step(
        "Layer 4b: POD spectrum",
        [PY, "scripts/viz_pod_spectrum.py",
         "--basis", str(basis_file),
         "--K", "8",
         "--out", str(out_dir / "layer4_spectrum.png")],
        out_dir / "layer4_spectrum.txt")
    # mode atlas is a nice-to-have, don't fail the layer if it errors
    _step(
        "Layer 4c: POD mode atlas (top 8)",
        [PY, "scripts/viz_pod_mode_atlas.py",
         "--basis", str(basis_file),
         "--K", "8",
         "--out", str(out_dir / "layer4_mode_atlas.png")],
        out_dir / "layer4_mode_atlas.txt")
    return ok_spec, f"{tail_train}\n---\n{tail_spec}"


def layer5_trial_train(npz_dir: Path, out_dir: Path, tag: str,
                         nx: int, ny: int, nt: int,
                         trial_limit: int) -> tuple[bool, str]:
    trial_tag = f"onboard_{tag}_trial"
    ok, _, tail = _step(
        "Layer 5: trial train (K=8, 50 epochs, 1000 sims)",
        [PY, "scripts/train.py",
         "--config", "configs/default.yaml",
         "--data.npz_dir", str(npz_dir),
         "--data.nx", str(nx), "--data.ny", str(ny),
         "--data.nt", str(nt),
         "--data.limit", str(trial_limit),
         "--pod.k", "8",
         "--train.epochs", "50",
         "--seeds", "[7]",
         "--tag", trial_tag],
        out_dir / "layer5_trial.txt")
    return ok, tail


def _write_report(out_dir: Path, results: list[dict],
                    npz_dir: Path, tag: str) -> None:
    lines = [f"# Onboarding report: {tag}\n",
             f"Dataset: `{npz_dir}`\n",
             f"Report dir: `{out_dir}`\n\n",
             "## Layer summary\n\n",
             "| # | Layer | Status | Duration | Notes |\n",
             "|---|---|---|---|---|\n"]
    for r in results:
        status = "OK" if r["ok"] else ("SKIPPED" if r.get("skipped")
                                          else "FAIL")
        note = "" if r["ok"] else r.get("note", "see log")
        lines.append(
            f"| {r['n']} | {r['name']} | {status} | "
            f"{r.get('duration', '--')} | {note} |\n")
    lines.append("\n## Verdict\n\n")
    first_fail = next((r for r in results
                        if not r["ok"] and not r.get("skipped")), None)
    if first_fail is None:
        lines.append("All 6 layers passed. Dataset is ready for "
                      "full training / sensor sweep.\n\n"
                      "Next: run `run_sweep.py` after adjusting "
                      "NPZ_DIR, or launch a plain full train via "
                      "`run_overnight.py`.\n")
    else:
        lines.append(
            f"Stopped at **Layer {first_fail['n']}** "
            f"({first_fail['name']}). Inspect the log at "
            f"`{first_fail.get('log', 'unknown')}` and address "
            "before re-running. Subsequent layers were skipped.\n")
    lines.append("\n## Artifacts to inspect\n\n"
                  "- `layer1_gt_quality/` -- per-sim quality "
                  "metrics + summary.json\n"
                  "- `layer2_diversity.png` -- per-cell std across "
                  "sims (empty regions = POD blind spots)\n"
                  "- `layer2_time_density.png` -- native time "
                  "sampling density\n"
                  "- `layer2_showcase/` -- per-sim visualizations "
                  "(top-down GIF, 5-angle radial kymograph, 3D "
                  "snapshot strip) for the first N sims; eyeball "
                  "what a typical sim in this dataset looks like\n"
                  "- `layer4_spectrum.png` -- POD energy decay; "
                  "check cumulative energy at K=8\n"
                  "- `layer4_mode_atlas.png` -- top 8 mode shapes "
                  "for physical intuition\n"
                  "- `layer5_trial.txt` -- final median field err "
                  "+ gap_to_floor from the trial run\n")
    (out_dir / "REPORT.md").write_text("".join(lines))
    print(f"\n=== REPORT: {out_dir / 'REPORT.md'} ===", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz-dir", required=True,
                    help="path to the new NPZ dataset")
    ap.add_argument("--tag", required=True,
                    help="short dataset name (used for report dir, "
                    "trial-train tag, etc.)")
    ap.add_argument("--out-root", default="viz",
                    help="parent dir under which onboard_<tag>/ "
                    "is created (default: viz)")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=128)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--gt-quality-n", type=int, default=200,
                    help="sim count for Layer 1 GT quality sample "
                    "(default: 200)")
    ap.add_argument("--showcase-n", type=int, default=3,
                    help="sim count for Layer 2c per-sim viz "
                    "showcase (top-down GIF + radial kymograph "
                    "+ 3D strip). Deterministic first-N pick by "
                    "sorted filename. Default: 3.")
    ap.add_argument("--probe-limit", type=int, default=300,
                    help="sim count for Layer 4 POD probe "
                    "(default: 300)")
    ap.add_argument("--trial-limit", type=int, default=1000,
                    help="sim count for Layer 5 trial train "
                    "(default: 1000)")
    ap.add_argument("--skip-trial", action="store_true",
                    help="stop after Layer 4 (POD probe); useful "
                    "when you just want data health, not model "
                    "health")
    ap.add_argument("--start-layer", type=int, default=0,
                    help="skip layers with index < START_LAYER "
                    "(useful when resuming after fixing an issue "
                    "-- earlier layers already succeeded)")
    ap.add_argument("--only-showcase", action="store_true",
                    help="run ONLY the Layer 2c per-sim showcase "
                    "(top-down GIF + radial kymograph + 3D strip). "
                    "Skips every other layer. Use when the rest "
                    "of onboarding already succeeded and you just "
                    "want the sim visualizations added.")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir).expanduser().resolve()
    if not npz_dir.is_dir():
        print(f"npz_dir not found: {npz_dir}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_root) / f"onboard_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Onboarding {args.tag} from {npz_dir}", flush=True)
    print(f"Report dir: {out_dir}", flush=True)

    if args.only_showcase:
        print(f"[--only-showcase] Running only Layer 2c on "
              f"{args.showcase_n} sims", flush=True)
        t0 = time.time()
        ok, tail = _layer2c_sim_showcase(
            npz_dir, out_dir, args.nx, args.ny, args.nt,
            args.showcase_n)
        dt = time.time() - t0
        print(f"\nshowcase done in {dt / 60:.1f} min "
              f"-- ok={ok}\n{tail}", flush=True)
        print(f"Artifacts in: {out_dir / 'layer2_showcase'}",
              flush=True)
        return 0 if ok else 1

    layers = [
        (0, "Basic sanity", lambda: layer0_inspect_npz(
            npz_dir, out_dir)),
        (1, "GT quality", lambda: layer1_gt_quality(
            npz_dir, out_dir, args.gt_quality_n)),
        (2, "Data diversity + sim showcase",
         lambda: layer2_diversity(
            npz_dir, out_dir, args.nx, args.ny, args.nt,
            args.showcase_n)),
        (3, "Loader smoke", lambda: layer3_loader_smoke(
            npz_dir, out_dir, args.nx, args.ny, args.nt)),
        (4, "POD completeness", lambda: layer4_pod_probe(
            npz_dir, out_dir, args.tag,
            args.nx, args.ny, args.nt,
            args.probe_limit, k_cache=32)),
    ]
    if not args.skip_trial:
        layers.append((5, "Trial train", lambda: layer5_trial_train(
            npz_dir, out_dir, args.tag,
            args.nx, args.ny, args.nt, args.trial_limit)))

    results: list[dict] = []
    all_ok = True
    t_start = time.time()
    for n, name, fn in layers:
        if n < args.start_layer:
            results.append(dict(n=n, name=name, ok=True,
                                 skipped=True,
                                 duration=f"skipped (< "
                                 f"start-layer {args.start_layer})"))
            print(f"\nSkipping Layer {n} ({name}) "
                  f"[--start-layer={args.start_layer}]",
                  flush=True)
            continue
        if not all_ok:
            results.append(dict(n=n, name=name, ok=False,
                                 skipped=True,
                                 duration="skipped"))
            continue
        t0 = time.time()
        try:
            ok, tail = fn()
        except Exception as e:
            ok = False
            tail = f"{type(e).__name__}: {e}"
        results.append(dict(
            n=n, name=name, ok=ok,
            duration=f"{time.time() - t0:.0f}s",
            note=("" if ok else tail.replace("\n", " | ")[:120])))
        if not ok:
            all_ok = False
            print(f"\nAborting: Layer {n} ({name}) FAILED. "
                  "Fix and re-run.", flush=True)

    _write_report(out_dir, results, npz_dir, args.tag)
    print(f"\nTotal duration: {(time.time() - t_start) / 60:.1f} min",
          flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
