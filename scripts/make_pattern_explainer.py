"""Build a self-contained HTML presenter guide that explains, for a
non-expert audience, how the bonding patterns were found and how to
read Figure 2 and Figure 3.

Point it at the pattern-analysis figure directory. It embeds Figure 1
(clean two-axes) and the ANNOTATED Figure 2 / Figure 3 (from
annotate_pattern_figures.py) as base64 so the page is one portable
file, and wraps each with a plain-language method blurb, an everyday
analogy, and a "how to read this" caption. Zero math notation.

    python scripts/make_pattern_explainer.py \\
        --fig-dir viz/pattern_analysis \\
        --out viz/pattern_analysis/explainer.html
"""
from __future__ import annotations
import argparse
import html
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.make_placement_report import _embed           # noqa: E402


_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,
sans-serif;max-width:960px;margin:2rem auto;padding:0 1.2rem;
color:#1c2733;line-height:1.65;}
h1{color:#14263b;margin-bottom:.2em;}
.sub{color:#5a6a7a;margin-top:0;}
h2{color:#1a3350;border-bottom:1px solid #d7dee7;padding-bottom:.3em;
margin-top:2.6em;}
.method{margin:.6em 0 1em;}
.analogy{background:#eef7f4;border-left:4px solid #2a9d8f;
padding:.7em 1em;border-radius:4px;margin:1em 0;}
.analogy b{color:#1f7a6c;}
figure{margin:1.4em 0;text-align:center;}
img{max-width:100%;height:auto;border:1px solid #e2e8f0;
border-radius:4px;}
figcaption{color:#5a6a7a;font-size:.92em;margin-top:.5em;
text-align:left;}
.read{background:#fff6da;border-left:4px solid #caa94a;
padding:.7em 1em;border-radius:4px;margin:1em 0;font-size:.95em;}
.read b{color:#9a7d1e;}
.pill{display:inline-block;background:#e7eef7;border-radius:12px;
padding:.12em .8em;font-size:.85em;margin:.2em .3em .2em 0;}
.takeaway{background:#f3f0fb;border:1px solid #d8cef0;
border-radius:6px;padding:1em 1.2em;margin:2em 0;}
.takeaway h2{border:0;margin-top:0;color:#4b3b8f;}
"""


def _fig(uri, title, caption_html):
    if not uri:
        return (f"<p><i>(missing figure: {html.escape(title)})</i>"
                f"</p>")
    return (f"<figure><img src=\"{uri}\" alt=\"{html.escape(title)}\">"
            f"<figcaption>{caption_html}</figcaption></figure>")


def _analogy(text_html):
    return f"<div class=\"analogy\"><b>Everyday analogy.</b> " \
           f"{text_html}</div>"


def _read(text_html):
    return f"<div class=\"read\"><b>How to read it.</b> " \
           f"{text_html}</div>"


def build_html(title, fig1, fig2, fig3):
    P = [f"<!doctype html><html><head><meta charset=\"utf-8\">"
         f"<title>{html.escape(title)}</title>"
         f"<style>{_CSS}</style></head><body>"]
    P.append(f"<h1>{html.escape(title)}</h1>")
    P.append("<p class=\"sub\">A plain-language guide to how we found "
             "the bonding patterns in the simulation data, and how to "
             "read the two key figures. No math required.</p>")
    P.append("<p><span class=\"pill\">two independent axes</span>"
             "<span class=\"pill\">time: release timing</span>"
             "<span class=\"pill\">space: symmetric vs "
             "asymmetric</span></p>")

    # --- Section 1: time patterns ---
    P.append("<h2>1. The three time patterns</h2>")
    P.append(
        "<p class=\"method\">The bonded region grows over time. We "
        "boil each simulation down to a single <b>progress gauge</b> "
        "-- the strength of the dominant shared shape as it evolves, "
        "rescaled from 0 (start) to 1 (done). A simple grouping "
        "algorithm then sorts these progress curves into look-alike "
        "families. We never label them by hand; the data itself "
        "supported <b>three</b> families, which we name by how fast "
        "they start: <b>direct release</b> (rises immediately) and "
        "two <b>hold-then-release</b> styles (pause first, then "
        "rise).</p>")
    P.append(_analogy(
        "Like watching runners' pace curves. Some sprint from the "
        "gun (direct); others wait, then go (hold). We don't tell "
        "the computer who is who -- it groups the pace curves into "
        "styles, and the number of real styles is whatever the data "
        "supports."))
    P.append(_fig(
        fig1, "Two axes",
        "<b>Figure 1.</b> Left: the average progress curve of each "
        "time family (direct vs hold). Middle: how one-sided each "
        "simulation is, for the old vs new dataset, with the "
        "symmetric/asymmetric cutoff. Right: how many simulations "
        "fall into each combined pattern."))

    # --- Section 2: sym vs asym ---
    P.append("<h2>2. Symmetric vs asymmetric</h2>")
    P.append(
        "<p class=\"method\">A <b>symmetric</b> bond looks the same "
        "in every direction -- its shape depends only on distance "
        "from the center, like a bullseye. An <b>asymmetric</b> bond "
        "is lopsided -- its shape also depends on direction, one "
        "side differing from another. The building-block shapes come "
        "in these two flavors: <b>radial</b> (rings) and "
        "<b>angle-varying</b> (lopsided). For each simulation we "
        "measure the share of its energy sitting in the angle-varying "
        "shapes. Even a hair of it counts as asymmetric -- that is a "
        "deliberate physical choice, set as a small fixed cutoff.</p>")
    P.append(_analogy(
        "Like ripples in a pond. A stone dropped dead-center makes "
        "perfect rings (symmetric -- only distance matters). "
        "Off-center, or a tilted pond, makes the ripples bunch to "
        "one side (asymmetric -- direction now matters). We simply "
        "measure how one-sided the wave is."))

    # --- Section 3: figure 2 ---
    P.append("<h2>3. Reading Figure 2 -- which shapes distinguish the "
             "patterns</h2>")
    P.append(_analogy(
        "Like a graphic equalizer. Each pattern is a song and each "
        "building-block shape is a frequency band. The bottom panel "
        "shows which bands each song pushes <b>above average</b> -- "
        "and the asymmetric 'songs' crank the high, angle-varying "
        "bands that the symmetric ones leave flat."))
    P.append(_fig(
        fig2, "Figure 2 annotated",
        "<b>Figure 2.</b> Rows are the patterns; columns are the "
        "building-block shapes (m1, m2, ...). The boxed columns are "
        "the angle-varying (azimuthal) shapes."))
    P.append(_read(
        "The <b>top</b> panel is the raw energy each pattern spends "
        "on each shape, on a log scale (the first shape holds ~90%, "
        "so a plain scale would hide the rest). The <b>bottom</b> "
        "panel divides each column by its average across patterns, so "
        "<b>red means this pattern over-uses that shape</b>. Look at "
        "the boxed columns: symmetric rows stay pale/blue, asymmetric "
        "rows turn red. That is where the lopsided patterns live."))

    # --- Section 4: figure 3 ---
    P.append("<h2>4. Reading Figure 3 -- why more shapes (K) "
             "helped</h2>")
    P.append(
        "<p class=\"method\">Our compression keeps only a fixed "
        "number of building-block shapes (K) and discards the rest. "
        "Whatever is discarded is an error nothing downstream can "
        "undo -- the <b>floor</b>. This figure shows that floor for "
        "each pattern at K=8 vs K=12.</p>")
    P.append(_analogy(
        "Like sketching a face from a fixed kit of template "
        "features. Eight templates nail a plain, symmetric face. A "
        "lopsided face needs a couple of extra templates -- and the "
        "leftover sketch error drops sharply once you add them. Plain "
        "faces gain nothing from the extras; lopsided ones do."))
    P.append(_fig(
        fig3, "Figure 3 annotated",
        "<b>Figure 3.</b> Two bars per pattern: the leftover error "
        "(floor) at K=8 and at K=12. Lower is better."))
    P.append(_read(
        "The <b>drop</b> from the K=8 bar to the K=12 bar is how much "
        "the four extra shapes help that pattern. <b>Symmetric bars "
        "barely move</b> -- K=8 already captures them. <b>Asymmetric "
        "bars fall sharply</b> -- the extra shapes are exactly what "
        "they needed. That is the concrete payoff of going from K=8 "
        "to K=12."))

    # --- takeaway ---
    P.append("<div class=\"takeaway\"><h2>The one-sentence "
             "takeaway</h2>"
             "<p>Time-of-release and left-right symmetry are two "
             "independent knobs on the data; the asymmetric patterns "
             "live in the higher, angle-varying shapes, so adding "
             "those shapes (K=8 to K=12) improved the reconstruction "
             "specifically for them.</p></div>")

    P.append("</body></html>")
    return "".join(P)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fig-dir", default="viz/pattern_analysis",
                    help="dir with 01_two_axes.png and the annotated "
                    "02_/03_ figures")
    ap.add_argument("--out", default=None,
                    help="output HTML (default <fig-dir>/"
                    "explainer.html)")
    ap.add_argument("--title",
                    default="How we found the bonding patterns")
    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    out = Path(args.out) if args.out else fig_dir / "explainer.html"

    fig1 = _embed(fig_dir / "01_two_axes.png")
    fig2 = _embed(fig_dir / "02_pattern_mode_occupancy_annotated.png")
    fig3 = _embed(fig_dir / "03_floor_by_pattern_annotated.png")
    if fig2 is None or fig3 is None:
        print("annotated figures missing; run "
              "annotate_pattern_figures.py first", file=sys.stderr)

    doc = build_html(args.title, fig1, fig2, fig3)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
