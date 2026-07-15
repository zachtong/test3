"""Build a self-contained HTML report on the sensor-placement study
and overall project status, and save every embedded figure ALSO as
a standalone PDF for reuse in a paper.

The report figures live on the machine that ran the experiments, so
this runs THERE: point it at the PNGs you already generated plus the
diagnose_worst_cases --out-json comparison, and it produces:

    <out-dir>/report.html          self-contained (figures base64
                                   embedded), open or print to PDF
    <out-dir>/pdf/<name>.pdf       one standalone PDF per figure

Every figure argument is optional -- missing ones are skipped, so
run it with whatever you have. The differentiable-vs-ABCDEF
comparison table + bar chart are rendered from --compare-json.

    python scripts/make_placement_report.py \\
        --out-dir viz/placement_report \\
        --compare-json viz/report_data/diffplace_vs_abcdef.json \\
        --fig-k-vs-sensor viz/merged_sweep_k12_summary/... \\
        --fig-diminishing viz/merged_sweep_k12_summary/diminishing_returns.png \\
        --fig-importance  viz/merged_sweep_k12_summary/sensor_importance.png \\
        --fig-qrdeim      viz/qrdeim_n6_k12.png \\
        --fig-diffplace   viz/diffplace_n6_k12.png \\
        --extra "POD mode atlas (K=12):viz/mode_atlas_k12.png"
"""
from __future__ import annotations
import argparse
import base64
import html
import json
import sys
from pathlib import Path


# --- figure helpers -------------------------------------------------

def _embed(png_path) -> str | None:
    p = Path(png_path)
    if not p.is_file():
        print(f"  skip (missing): {p}", file=sys.stderr)
        return None
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    ext = p.suffix.lstrip(".").lower()
    mime = "png" if ext == "png" else ("jpeg" if ext in
                                       ("jpg", "jpeg") else ext)
    return f"data:image/{mime};base64,{data}"


def _png_to_pdf(png_path, pdf_path) -> bool:
    """Wrap a raster PNG into a standalone PDF at native resolution."""
    try:
        from PIL import Image
        im = Image.open(png_path).convert("RGB")
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        im.save(pdf_path, "PDF", resolution=150.0)
        return True
    except Exception:
        pass
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img = plt.imread(str(png_path))
        h, w = img.shape[:2]
        fig = plt.figure(figsize=(w / 150.0, h / 150.0))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.axis("off")
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(pdf_path))
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  PDF conversion failed for {png_path}: {e}",
              file=sys.stderr)
        return False


def _render_comparison(compare, png_path, pdf_path):
    """Grouped bar chart of median / p95 / worst-N field per tag,
    from a diagnose_worst_cases --out-json. Saved as PNG + PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tags = compare.get("tags", [])
    if not tags:
        return None
    names = [t["tag"] for t in tags]
    scale = 100.0                         # relative L2 -> percent
    metrics = [("median", "median"),
               ("p95", "p95"),
               ("worst_n_field",
                f"worst-{compare.get('top_n', 20)} mean")]
    x = np.arange(len(metrics))
    w = 0.8 / max(len(names), 1)
    palette = ["#3d5a80", "#e63946", "#2a9d8f", "#e9c46a"]
    fig, ax = plt.subplots(figsize=(8.5, 5.0),
                           constrained_layout=True)
    for i, t in enumerate(names):
        vals = [(tags[i].get(k) or 0.0) * scale for k, _ in metrics]
        bars = ax.bar(x + i * w, vals, w, label=t,
                      color=palette[i % len(palette)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8)
    ax.set_xticks(x + w * (len(names) - 1) / 2)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("field error (%)")
    ax.set_title("Differentiable-optimized placement vs ABCDEF")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    return _embed(png_path)


# --- report text ----------------------------------------------------

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,
sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;
color:#1c2733;line-height:1.6;}
h1{color:#14263b;} h2{color:#1a3350;border-bottom:1px solid #d7dee7;
padding-bottom:.3em;margin-top:2.4em;} h3{color:#26456b;}
figure{margin:1.4em 0;text-align:center;}
img{max-width:100%;height:auto;border:1px solid #e2e8f0;
border-radius:4px;}
figcaption{color:#5a6a7a;font-size:.9em;margin-top:.5em;}
table{border-collapse:collapse;margin:1em 0;font-size:.93em;}
th,td{border:1px solid #cfd8e3;padding:.4em .8em;text-align:left;}
th{background:#eef3f8;}
.key{background:#fff6da;padding:.15em .4em;border-radius:3px;}
.pill{display:inline-block;background:#e7eef7;border-radius:12px;
padding:.1em .7em;font-size:.85em;margin-right:.3em;}
code{background:#f1f4f8;padding:.1em .3em;border-radius:3px;}
"""


def _fig_html(title, caption, data_uri):
    if not data_uri:
        return ""
    return (f"<figure><img src=\"{data_uri}\" alt=\""
            f"{html.escape(title)}\">"
            f"<figcaption><b>{html.escape(title)}.</b> "
            f"{html.escape(caption)}</figcaption></figure>\n")


def _comparison_table(compare) -> str:
    tags = compare.get("tags", [])
    if not tags:
        return ""
    tn = compare.get("top_n", 20)
    rows = ["<table><thead><tr><th>config</th><th>median</th>"
            "<th>p95</th><th>gap_to_floor</th>"
            f"<th>worst-{tn} mean</th></tr></thead><tbody>"]
    for t in tags:
        def f(k):
            v = t.get(k)
            return "--" if v is None else f"{v * 100:.3f}%" \
                if k != "gap_to_floor" else f"{v:.3f}"
        rows.append(
            f"<tr><td><code>{html.escape(t['tag'])}</code></td>"
            f"<td>{f('median')}</td><td>{f('p95')}</td>"
            f"<td>{f('gap_to_floor')}</td>"
            f"<td>{f('worst_n_field')}</td></tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def build_report(args, figs, compare, compare_uri) -> str:
    K = args.k
    P = []
    P.append(f"<!doctype html><html><head><meta charset=\"utf-8\">"
             f"<title>{html.escape(args.title)}</title>"
             f"<style>{_CSS}</style></head><body>")
    P.append(f"<h1>{html.escape(args.title)}</h1>")
    P.append(f"<p class=\"pill\">POD + BiTCN</p>"
             f"<p class=\"pill\">K = {K}</p>"
             f"<p class=\"pill\">3D non-axisymmetric wafer bonding"
             f"</p>")

    # 1. Executive summary
    P.append("<h2>1. Executive summary</h2>")
    P.append(
        "<p>Sensor-placement optimization for the 3D wafer-bonding "
        "reconstruction is <span class=\"key\">essentially "
        "solved</span>. Three independent methods -- an exhaustive "
        "subset sweep, an observability-based selection (QR-DEIM / "
        "observability Gramian), and gradient-based differentiable "
        "placement -- all converge on the same conclusion: the "
        "physically-motivated fixed layout (two rings x three "
        "angles, 'ABCDEF') is at or near optimal. Reconstruction "
        f"quality is bounded by the POD truncation (K={K}) and by "
        "sensor observability of the moving bonding front, not by "
        "fine placement.</p>")

    # 2. Project status
    P.append("<h2>2. Project status</h2>")
    P.append(
        "<p>The pipeline reconstructs the full upper-wafer "
        "displacement field u_z(x, y, t) over a quarter disk from a "
        "handful of point sensors: a POD basis compresses the field "
        f"to K={K} spatial modes, and a bi-directional temporal "
        "convolutional network (BiTCN) maps the sensor time series "
        "to the modal coefficient trajectories, from which the "
        "field is reconstructed.</p>")
    P.append("<h3>Data</h3>")
    P.append(
        "<p>Simulations from a 3D non-axisymmetric COMSOL model. "
        "Work proceeded on an initial large batch, a second smaller "
        "batch that reproduced its conclusions, and a merged "
        "dataset (symlink union) used for the final placement "
        "study. The loader canonicalizes each native COMSOL point "
        "cloud onto a common quarter-disk Cartesian grid per "
        "timestep, then resamples to a uniform canonical time "
        "axis.</p>")
    P.append("<h3>POD dimensionality (K)</h3>")
    P.append(
        f"<p>K={K} captures ~99.99% of the field energy; the model "
        "reaches within ~10% of the resulting POD truncation floor "
        "(median gap-to-floor ~1.1). A K sweep showed absolute "
        "worst-case error keeps dropping with K while the gap to "
        "floor widens -- reconstruction is limited by sensor "
        "OBSERVABILITY of the higher modes, not by the modes' "
        "absence. K=12 is the practical sweet spot.</p>")
    P.append("<h3>A physics correction: trapped-gas bulges</h3>")
    P.append(
        "<p>During bonding, gas trapped behind the moving front is "
        "expelled outward as a propagating bulge that transiently "
        "lifts the upper wafer (u_z rises then descends again at a "
        "fixed location). An earlier loader step mistook this for a "
        "step-boundary artifact and smoothed it away; the loader "
        "was corrected to preserve it, since this rebound is real "
        "physics and is essential for downstream gas-trap anomaly "
        "detection.</p>")

    # 3. Placement study
    P.append("<h2>3. Sensor-placement study</h2>")

    P.append("<h3>3.1 K dominates; placement is second-order</h3>")
    P.append(
        "<p>Across every sensor configuration at a fixed K, the "
        "median field error clusters tightly; the cross-K gap "
        "dwarfs the within-K spread. Once basic angle and radius "
        "coverage is met, the choice of sensor positions is a "
        "second-order effect and K is the first-order lever.</p>")
    P.append(_fig_html(
        "K dominates, placement secondary",
        "Per-config field error grouped by K; tight boxes far apart "
        "= K is the lever, placement is not.",
        figs.get("k_vs_sensor")))

    P.append("<h3>3.2 How many sensors, and where</h3>")
    P.append(
        "<p>A subset sweep over the fixed six-position catalogue "
        "(two rings x three angles) showed diminishing returns: a "
        "small number of sensors already reaches the floor, and "
        "sensor importance ranks the outer ring above the inner, "
        "and mixed-ring above single-ring -- consistent with the "
        "need to observe the front's radial propagation over "
        "time.</p>")
    P.append(_fig_html(
        "Diminishing returns vs sensor count",
        "Field error vs number of sensors; the curve flattens "
        "quickly.", figs.get("diminishing")))
    P.append(_fig_html(
        "Sensor importance",
        "Frequency in top configs and n-controlled impact per "
        "sensor.", figs.get("importance")))

    P.append("<h3>3.3 Instantaneous observability is not enough "
             "(QR-DEIM)</h3>")
    P.append(
        "<p>QR-DEIM places sensors to maximize instantaneous "
        "spatial observability of the POD modes. It clustered most "
        "sensors on the outer rim (where the modes are angularly "
        "most distinctive) and UNDER-performed the fixed ABCDEF "
        "layout. The reason: reconstruction uses the sensor time "
        "series and the front's propagation; an instantaneous "
        "criterion ignores the radial/temporal coverage that "
        "ABCDEF's two-ring spread provides.</p>")
    P.append(_fig_html(
        "QR-DEIM optimal vs ABCDEF",
        "QR-DEIM (circles) clusters on the outer rim; ABCDEF "
        "(squares) spreads across two radii.",
        figs.get("qrdeim")))

    P.append("<h3>3.4 Differentiable placement (the principled "
             "test)</h3>")
    P.append(
        "<p>To optimize placement against the true reconstruction "
        "objective rather than a proxy, the sensor coordinates were "
        "made continuous learnable parameters and the measurement "
        "made differentiable (bilinear interpolation of the modes "
        "at the sensor position), so the reconstruction loss "
        "gradient flows to the coordinates. Initialized from "
        "ABCDEF and refined within the feasible band, the sensors "
        "moved <span class=\"key\">purely radially, with zero "
        "azimuthal motion</span>, and converged by ~250 epochs. "
        "Angular placement (0/45/90 deg) is already saturated under "
        "quarter symmetry; only a slight radial spread is favored.</p>")
    P.append(_fig_html(
        "Differentiable placement",
        "Sensor movement paths from ABCDEF init, per-sensor "
        "movement-vs-epoch convergence, and loss.",
        figs.get("diffplace")))

    P.append("<h3>3.5 Verification: no meaningful gap</h3>")
    if compare:
        P.append(
            "<p>The differentiably-refined positions were retrained "
            "through the standard pipeline and scored against "
            "ABCDEF on held-out test sims. The worst-case field "
            "error is essentially identical -- the small radial "
            "adjustment buys nothing measurable, confirming ABCDEF "
            "is at (or near) the optimum.</p>")
        P.append(_comparison_table(compare))
        P.append(_fig_html(
            "Optimized placement vs ABCDEF",
            "Median, p95, and worst-N mean field error; the bars "
            "are effectively equal.", compare_uri))
    else:
        P.append("<p>(Comparison data not supplied; pass "
                 "<code>--compare-json</code>.)</p>")

    # extras
    if figs.get("extras"):
        P.append("<h2>4. Additional figures</h2>")
        for title, uri in figs["extras"]:
            P.append(_fig_html(title, "", uri))

    # conclusion + next
    P.append("<h2>5. Conclusion and next directions</h2>")
    P.append(
        "<p><b>Placement is closed.</b> The physically-motivated "
        "ABCDEF layout is near-optimal, verified by gradient-based "
        "optimization; angular placement is saturated and radial "
        "spread is the only, minor, lever. Further placement "
        "sweeps are not warranted.</p>")
    P.append(
        "<p><b>Open frontiers (not placement):</b> (i) anomaly / "
        "defect detection -- train on normal data, flag deviations, "
        "with the now-preserved gas-bulge physics enabling gas-trap "
        "detection; (ii) evaluation on real experimental sensor "
        "data (sim-to-real) via a self-contained inference bundle; "
        "(iii) increasing K only if the truncation floor ever needs "
        "to be lowered.</p>")

    P.append("</body></html>")
    return "".join(P)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="viz/placement_report")
    ap.add_argument("--title",
                    default="Sensor Placement and Project Status")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--compare-json", default=None,
                    help="diagnose_worst_cases --out-json output")
    ap.add_argument("--fig-k-vs-sensor", default=None)
    ap.add_argument("--fig-diminishing", default=None)
    ap.add_argument("--fig-importance", default=None)
    ap.add_argument("--fig-qrdeim", default=None)
    ap.add_argument("--fig-diffplace", default=None)
    ap.add_argument("--extra", action="append", default=[],
                    help="'Title:path.png' extra figure(s); "
                    "repeatable")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pdf_dir = out_dir / "pdf"
    out_dir.mkdir(parents=True, exist_ok=True)

    # embed known figures + write their standalone PDFs
    figs = {}
    known = [("k_vs_sensor", args.fig_k_vs_sensor),
             ("diminishing", args.fig_diminishing),
             ("importance", args.fig_importance),
             ("qrdeim", args.fig_qrdeim),
             ("diffplace", args.fig_diffplace)]
    for key, path in known:
        if not path:
            continue
        uri = _embed(path)
        if uri:
            figs[key] = uri
            if _png_to_pdf(path, pdf_dir / f"{key}.pdf"):
                print(f"  pdf: {pdf_dir / (key + '.pdf')}")

    extras = []
    for spec in args.extra:
        if ":" not in spec:
            print(f"  bad --extra (need Title:path): {spec}",
                  file=sys.stderr)
            continue
        title, path = spec.split(":", 1)
        uri = _embed(path.strip())
        if uri:
            extras.append((title.strip(), uri))
            safe = "".join(c if c.isalnum() else "_"
                           for c in title.strip())[:40]
            _png_to_pdf(path.strip(), pdf_dir / f"extra_{safe}.pdf")
    if extras:
        figs["extras"] = extras

    # comparison figure from json (rendered here -> PNG + PDF)
    compare = None
    compare_uri = None
    if args.compare_json and Path(args.compare_json).is_file():
        compare = json.loads(Path(args.compare_json).read_text())
        compare_uri = _render_comparison(
            compare, out_dir / "_compare.png",
            pdf_dir / "comparison.pdf")

    html_str = build_report(args, figs, compare, compare_uri)
    report = out_dir / "report.html"
    report.write_text(html_str)
    print(f"\nwrote {report}")
    print(f"standalone PDFs in {pdf_dir}/")
    n_fig = len([k for k in figs if k != "extras"]) + \
        len(figs.get("extras", [])) + (1 if compare_uri else 0)
    print(f"embedded {n_fig} figure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
