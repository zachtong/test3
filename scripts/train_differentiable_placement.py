"""Differentiable sensor placement: optimize sensor (r, theta) by
gradient descent jointly with the reconstruction model, instead of
searching over discrete add/remove-a-sensor candidates.

Why this is the principled method: the true objective is
"minimize the model's reconstruction error as a function of sensor
positions." The sweep evaluates that objective by RETRAINING for
every candidate placement -- combinatorial, and it only ranks
candidates, never tells you which DIRECTION improves. Here the
positions are continuous learnable parameters and the measurement
is differentiable, so one training run moves the sensors DOWN the
reconstruction-loss gradient. It targets the true (nonlinear,
trained-model) objective directly -- unlike QR-DEIM / observability
proxies, which we saw diverge from it.

Differentiable measurement (the key trick): the field is
f(x,y,t) ~ sum_k Phi_k(x,y) a(k,t). A sensor at position p measures
y(t) = f(p, t) ~ [Phi interpolated at p] . a(t). Bilinear
interpolation (torch grid_sample) is differentiable w.r.t. p, so
the gradient flows from the reconstruction loss all the way to the
sensor coordinates. Only Phi (Nx*Ny, K) and the modal trajectories
a (n_sim, K, Nt) are needed -- never the full 93 GB field.

Minimal version: fixed sensor count n, positions initialized from a
starting layout (default ABCDEF) and gradient-refined within the
feasible band [r-min, r-max] x [0, 90] deg. Answers directly:
is ABCDEF a local optimum, and if not, which way do the sensors
want to move.

    python scripts/train_differentiable_placement.py \\
        --basis outputs/basis_cache/pod3d_<key>.npz \\
        --npz-dir /data/dataset --K 12 --n 6 \\
        --init abcdef --epochs 300 \\
        --out viz/diffplace_n6_k12.png
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.grid import canonical_grid, disk_mask         # noqa: E402

_ABCDEF = [(0.52, 0.0), (0.52, 45.0), (0.52, 90.0),
           (0.847, 0.0), (0.847, 45.0), (0.847, 90.0)]


def _load_phi_and_a(basis_path, traj_path, npz_dir, K, nt,
                    drop_first_steps, limit):
    """Return Phi (Nx*Ny, K), a (n_sim, K, Nt), (nx, ny). a comes
    from a traj cache or is recomputed from a dataset subsample."""
    with np.load(basis_path) as z:
        Phi = z["Phi"]
        nx, ny = (int(d) for d in z["spatial_shape"])
    if traj_path:
        with np.load(traj_path, allow_pickle=False) as z:
            a = z["a_train_val"].astype(np.float64)
    elif npz_dir:
        from data.loader import load_dataset
        _x, _y, sims = load_dataset(
            npz_dir, nx=nx, ny=ny, nt=nt, limit=limit,
            drop_first_steps=drop_first_steps)
        if not sims:
            raise ValueError(f"no sims from {npz_dir}")
        nspace = Phi.shape[0]
        kk = Phi.shape[1]
        a = np.empty((len(sims), kk, nt), dtype=np.float64)
        for i, s in enumerate(sims):
            f = np.asarray(s.f, dtype=np.float64).reshape(nspace, -1)
            a[i] = Phi.T @ f
    else:
        raise ValueError("provide --traj or --npz-dir")
    k = min(K, Phi.shape[1], a.shape[1])
    return Phi[:, :k], a[:, :k, :], (nx, ny)


def _init_positions(init, n):
    if init == "abcdef":
        base = _ABCDEF
    elif init == "random":
        return None                                  # filled later
    else:
        # accept an inline JSON string [[r,theta],...] OR a path to
        # a JSON file with the same content
        try:
            base = json.loads(init)
        except (json.JSONDecodeError, ValueError):
            base = json.loads(Path(init).read_text())
    if len(base) < n:
        raise ValueError(f"init has {len(base)} < n={n} positions")
    base = base[:n]
    r = np.array([p[0] for p in base], dtype=np.float64)
    th = np.array([p[1] for p in base], dtype=np.float64)
    return r, th


def run(basis_path, *, traj_path=None, npz_dir=None, K=12, n=6,
        init="abcdef", r_min=0.2, r_max=0.98, epochs=300, lr=1e-3,
        pos_lr=2e-2, val_frac=0.2, seed=7, nt=300,
        drop_first_steps=1, limit=400, channels=64,
        dilations=(1, 2, 4, 8, 16, 32, 64), kernel=3,
        device=None) -> dict:
    import torch
    import torch.nn.functional as F
    from models.registry import create_model

    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    Phi, a_np, (nx, ny) = _load_phi_and_a(
        basis_path, traj_path, npz_dir, K, nt, drop_first_steps,
        limit)
    K = Phi.shape[1]
    n_sim, _, Nt = a_np.shape

    # split sims
    perm = rng.permutation(n_sim)
    n_val = max(1, int(round(val_frac * n_sim)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    # Phi as a (1, K, ny, nx) image for grid_sample. Phi[:,k] is
    # flattened (nx, ny) [x, y]; transpose to (ny, nx) [y, x] so
    # grid coord 0 = x (width), 1 = y (height).
    Phi_img = np.stack([Phi[:, k].reshape(nx, ny).T
                        for k in range(K)], axis=0)      # (K, ny, nx)
    Phi_img = torch.tensor(Phi_img[None], dtype=torch.float32,
                           device=dev)                   # (1,K,ny,nx)
    a_t = torch.tensor(a_np, dtype=torch.float32, device=dev)

    # target normalization: per-mode std (fixed), so all modes count
    a_std = a_t.std(dim=(0, 2), keepdim=True).clamp_min(1e-8)
    a_norm = a_t / a_std

    # positions
    ip = _init_positions(init, n)
    if ip is None:
        r0 = rng.uniform(r_min, r_max, n)
        th0 = rng.uniform(0, 90, n)
    else:
        r0, th0 = ip
    r_par = torch.tensor(r0, dtype=torch.float32, device=dev,
                         requires_grad=True)
    th_par = torch.tensor(th0, dtype=torch.float32, device=dev,
                          requires_grad=True)
    init_pos = np.stack([r0, th0], axis=1).copy()

    def measure(idx):
        """Differentiable sensor time series for sims `idx`.
        Returns y (B, n, Nt)."""
        x = r_par * torch.cos(th_par * np.pi / 180.0)
        y = r_par * torch.sin(th_par * np.pi / 180.0)
        gx = 2.0 * x - 1.0                               # [0,1]->[-1,1]
        gy = 2.0 * y - 1.0
        grid = torch.stack([gx, gy], dim=-1)[None, None]  # (1,1,n,2)
        phi_at = F.grid_sample(Phi_img, grid,
                               mode="bilinear",
                               padding_mode="border",
                               align_corners=True)        # (1,K,1,n)
        phi_at = phi_at[0, :, 0, :]                        # (K, n)
        ab = a_t[idx]                                      # (B,K,Nt)
        yb = torch.einsum("kn,bkt->bnt", phi_at, ab)       # (B,n,Nt)
        return yb

    # input normalization scale from init positions (fixed)
    with torch.no_grad():
        y0 = measure(torch.tensor(tr_idx, device=dev))
        y_std = y0.std().clamp_min(1e-8)

    model = create_model("bitcn", n_in=n, n_out=K,
                         channels=channels,
                         dilations=list(dilations), kernel=kernel,
                         dropout=0.0, causal=False).to(dev)
    opt = torch.optim.Adam(
        [{"params": model.parameters(), "lr": lr},
         {"params": [r_par, th_par], "lr": pos_lr}])

    tr_i = torch.tensor(tr_idx, device=dev)
    vl_i = torch.tensor(val_idx, device=dev)
    pos_hist = [init_pos]
    val_hist = []
    best_val = np.inf
    best_pos = init_pos

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        y = measure(tr_i) / y_std
        a_pred = model(y)                                  # (B,K,Nt)
        loss = F.mse_loss(a_pred, a_norm[tr_i])
        loss.backward()
        opt.step()
        # project positions back into the feasible band
        with torch.no_grad():
            r_par.clamp_(r_min, r_max)
            th_par.clamp_(0.0, 90.0)

        if (ep + 1) % max(1, epochs // 30) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                yv = measure(vl_i) / y_std
                vloss = float(F.mse_loss(model(yv),
                                         a_norm[vl_i]))
            cur = np.stack([r_par.detach().cpu().numpy(),
                            th_par.detach().cpu().numpy()], axis=1)
            pos_hist.append(cur.copy())
            val_hist.append((ep + 1, float(loss.detach()), vloss))
            if vloss < best_val:
                best_val, best_pos = vloss, cur.copy()
            print(f"  ep {ep + 1:4d}  train "
                  f"{float(loss.detach()):.4e}  "
                  f"val {vloss:.4e}", flush=True)

    final_pos = np.stack([r_par.detach().cpu().numpy(),
                          th_par.detach().cpu().numpy()], axis=1)
    moved = np.sqrt(((final_pos[:, 0] * np.cos(np.deg2rad(final_pos[:, 1]))
                      - init_pos[:, 0] * np.cos(np.deg2rad(init_pos[:, 1]))) ** 2
                     + (final_pos[:, 0] * np.sin(np.deg2rad(final_pos[:, 1]))
                        - init_pos[:, 0] * np.sin(np.deg2rad(init_pos[:, 1]))) ** 2))
    return dict(init_pos=init_pos, final_pos=final_pos,
                best_pos=best_pos, best_val=best_val,
                pos_hist=pos_hist, val_hist=val_hist, moved=moved,
                K=K, n=n, r_min=r_min, r_max=r_max, n_sim=n_sim,
                Nt=Nt)


def _render(res, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axl) = plt.subplots(
        1, 2, figsize=(13, 6.2),
        gridspec_kw=dict(width_ratios=[1.3, 1.0]),
        constrained_layout=True)
    th = np.linspace(0, 90, 200)
    ax.plot(np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th)),
            color="0.35", lw=2)
    ax.plot([0, 1.05], [0, 0], color="0.7", lw=1)
    ax.plot([0, 0], [0, 1.05], color="0.7", lw=1)
    for rb in (res["r_min"], res["r_max"]):
        ax.plot(rb * np.cos(np.deg2rad(th)),
                rb * np.sin(np.deg2rad(th)),
                color="#2a9d8f", lw=1, ls="-", alpha=0.6)

    def xy(p):
        return (p[:, 0] * np.cos(np.deg2rad(p[:, 1])),
                p[:, 0] * np.sin(np.deg2rad(p[:, 1])))

    ix, iy = xy(res["init_pos"])
    fx, fy = xy(res["final_pos"])
    # movement trails
    for k in range(res["n"]):
        ax.annotate("", xy=(fx[k], fy[k]), xytext=(ix[k], iy[k]),
                    arrowprops=dict(arrowstyle="->", color="0.5",
                                    lw=1.2, alpha=0.8))
    ax.scatter(ix, iy, s=120, marker="s", facecolor="none",
               edgecolor="0.4", linewidth=1.6, label="init (ABCDEF)")
    ax.scatter(fx, fy, s=200, marker="o", color="#e63946",
               edgecolor="black", linewidth=1.2, zorder=5,
               label="optimized")
    ax.set_xlim(-0.08, 1.15)
    ax.set_ylim(-0.08, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")
    ax.set_title(f"Differentiable placement (n={res['n']}, "
                 f"K={res['K']})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    vh = np.array(res["val_hist"])
    if vh.size:
        axl.plot(vh[:, 0], vh[:, 1], "-o", ms=3, label="train",
                 color="#3d5a80")
        axl.plot(vh[:, 0], vh[:, 2], "-o", ms=3, label="val",
                 color="#e63946")
        axl.set_xlabel("epoch")
        axl.set_ylabel("MSE (normalized coeffs)")
        axl.set_yscale("log")
        axl.set_title("loss")
        axl.legend(fontsize=9)
        axl.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basis", required=True)
    ap.add_argument("--traj", default=None)
    ap.add_argument("--npz-dir", default=None)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--init", default="abcdef",
                    help="'abcdef', 'random', or a JSON "
                    "[[r,theta],...] path")
    ap.add_argument("--r-min", type=float, default=0.2)
    ap.add_argument("--r-max", type=float, default=0.98)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-lr", type=float, default=2e-2,
                    help="learning rate for sensor positions "
                    "(default 2e-2; larger than model lr so "
                    "positions move meaningfully)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--positions-json", default=None)
    args = ap.parse_args()

    if not Path(args.basis).is_file():
        print(f"basis not found: {args.basis}", file=sys.stderr)
        return 2
    res = run(args.basis, traj_path=args.traj, npz_dir=args.npz_dir,
              K=args.K, n=args.n, init=args.init,
              r_min=args.r_min, r_max=args.r_max, epochs=args.epochs,
              lr=args.lr, pos_lr=args.pos_lr, val_frac=args.val_frac,
              seed=args.seed, nt=args.nt,
              drop_first_steps=args.drop_first_steps,
              limit=args.limit)

    print(f"\nfinal positions (best val={res['best_val']:.4e}):")
    print(f"  {'#':>2}  {'r_init':>7} {'th_init':>7}  ->  "
          f"{'r_opt':>7} {'th_opt':>7}  {'moved':>7}")
    for i in range(res["n"]):
        ip, fp = res["init_pos"][i], res["best_pos"][i]
        print(f"  {i + 1:>2}  {ip[0]:7.3f} {ip[1]:7.1f}  ->  "
              f"{fp[0]:7.3f} {fp[1]:7.1f}  {res['moved'][i]:7.3f}")
    pos_json = json.dumps([[round(float(r), 4), round(float(t), 2)]
                           for r, t in res["best_pos"]])
    print(f"\noptimized positions JSON:\n  {pos_json}")
    tot = float(res["moved"].sum())
    print(f"\ntotal movement (canonical units): {tot:.3f}")
    if tot < 0.02 * res["n"]:
        print("  -> positions barely moved: the init layout is at "
              "(or near) a local optimum for this objective.")
    else:
        print("  -> positions moved substantially: the init layout "
              "is NOT optimal; the arrows show the improving "
              "direction.")

    if args.positions_json:
        Path(args.positions_json).write_text(pos_json)
        print(f"wrote {args.positions_json}")
    if args.out:
        _render(res, Path(args.out))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
