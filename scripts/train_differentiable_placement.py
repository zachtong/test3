"""Differentiable sensor placement: optimize sensor positions by
gradient descent jointly with the reconstruction model, instead of
searching over discrete add/remove-a-sensor candidates.

Why this is the principled method: the true objective is
"minimize the model's reconstruction error as a function of sensor
positions." The sweep evaluates that objective by RETRAINING for
every candidate placement -- combinatorial, and it only ranks
candidates, never tells you which DIRECTION improves. Here the
positions are continuous learnable parameters and the measurement
is differentiable, so one training run moves the sensors DOWN the
reconstruction-loss gradient.

Differentiable measurement (the key trick): the field is
f(x,y,t) ~ sum_k Phi_k(x,y) a(k,t). A sensor at position p measures
y(t) = f(p, t) ~ [Phi interpolated at p] . a(t). Bilinear
interpolation (torch grid_sample) is differentiable w.r.t. p, so
the gradient flows from the reconstruction loss all the way to the
sensor coordinates. Only Phi (Nx*Ny, K) and the modal trajectories
a (n_sim, K, Nt) are needed -- never the full field.

PARAMETERIZATION MATTERS (this is a real conditioning trap). If the
sensor is parameterized as (r, theta_deg), one gradient step of the
optimizer moves it by ~lr in EACH parameter's own units. In physical
(Cartesian) space that is ~lr radially but only ~lr * (pi/180) * r
tangentially -- about 60x LESS angular exploration per step, purely
because theta is in degrees. Adam does not fix this: it normalizes
gradient MAGNITUDE, not the parameter's physical scale. So a
degrees-parameterized run can look like "theta barely moved / the
layout only wants radial change" when in truth the angular direction
was just explored ~60x more slowly. Three `--param` modes:

  cartesian  (default) -- optimize (x, y) directly, project back into
      the feasible band each step. Radial and tangential exploration
      are physically ISOTROPIC. Use this to judge whether angle
      genuinely matters.
  polar-rad  -- (r, theta) with theta in radians (tangential step
      ~lr*r, roughly balanced with radial).
  polar-deg  -- (r, theta_deg), the original throttled behavior; kept
      for A/B comparison against cartesian.

Note: sensors sitting exactly on a mirror axis (theta = 0 or 90) are
symmetry STATIONARY points -- dL/dtheta = 0 there by the wafer's x/y
mirror symmetry -- so they correctly do not drift in angle under ANY
parameterization. To test whether angle matters, initialize sensors
OFF the axes (e.g. --init uniform-outer places them at 0/18/.../90).

    python scripts/train_differentiable_placement.py \\
        --basis outputs/basis_cache/pod3d_<key>.npz \\
        --npz-dir /data/dataset --K 12 --n 6 \\
        --init uniform-outer --param cartesian --epochs 500 \\
        --out viz/diffplace_uniform_cart.png \\
        --anim-out viz/diffplace_uniform_cart.gif \\
        --save-history viz/diffplace_uniform_cart_hist.npz
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.grid import canonical_grid, disk_mask         # noqa: E402
from models.registry import create_model                # noqa: E402

_ABCDEF = [(0.52, 0.0), (0.52, 45.0), (0.52, 90.0),
           (0.847, 0.0), (0.847, 45.0), (0.847, 90.0)]


def _random_subsample_dir(npz_dir, limit, seed):
    """Symlink a random `limit` sims into a temp dir so the loader's
    first-N behavior samples a REPRESENTATIVE subset regardless of
    filename ordering. Critical for merged datasets whose sorted
    order groups all of one source first. Returns (tempdir_obj,
    path) -- keep tempdir_obj alive until loading is done."""
    import os
    import tempfile
    src = Path(npz_dir)
    files = sorted(p for p in src.glob("*.npz")
                   if not p.name.startswith("_"))
    if not files:
        raise ValueError(f"no npz in {npz_dir}")
    rng = np.random.default_rng(seed)
    n = min(limit, len(files))
    pick = rng.choice(len(files), size=n, replace=False)
    tmp = tempfile.TemporaryDirectory(prefix="diffplace_subsample_")
    for i in pick:
        f = files[int(i)].resolve()
        os.symlink(f, Path(tmp.name) / f.name)
    return tmp, tmp.name


def _load_phi_and_a(basis_path, traj_path, npz_dir, K, nt,
                    drop_first_steps, limit, random_subsample=False,
                    seed=7):
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
        tmp = None
        if random_subsample:
            tmp, load_dir = _random_subsample_dir(npz_dir, limit, seed)
            load_limit = None            # dir already holds exactly N
        else:
            load_dir, load_limit = npz_dir, limit
        try:
            _x, _y, sims = load_dataset(
                load_dir, nx=nx, ny=ny, nt=nt, limit=load_limit,
                drop_first_steps=drop_first_steps)
        finally:
            if tmp is not None:
                tmp.cleanup()
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


def _parse_positions(spec):
    """Parse an --init spec that is either inline JSON [[r,theta],...] OR a
    path to a JSON file with the same content. Fails with a CLEAR, actionable
    message rather than silently reinterpreting a mangled JSON string as a
    filename (which produced a confusing 'no such file' error)."""
    s = spec.strip()
    if s.startswith("["):
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"--init looks like inline JSON but did not parse ({e}). Your "
                f"shell most likely mangled the brackets/quotes (zsh treats "
                f"[ ] as globs, and pasting can turn ' into a smart quote) -- "
                f"write the positions to a file and pass --init <file.json>. "
                f"got: {spec[:80]!r}") from e
    p = Path(spec)
    if not p.is_file():
        raise ValueError(
            f"--init {spec!r} is not a preset (abcdef / uniform-outer / diag45 "
            f"/ random), not inline JSON (must start with '['), and not an "
            f"existing file. For inline positions the shell probably stripped "
            f"the quotes; put them in a file and pass its path.")
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"--init file {spec!r} is not valid JSON: {e}") from e


def _init_positions(init, n, r_min=0.2, r_max=0.98):
    if init == "abcdef":
        base = _ABCDEF
    elif init == "uniform-outer":
        # outer ring, angles spread across [0, 90] deg -- 4 of 6 land OFF the
        # mirror axes so they can feel a real tangential gradient.
        return (np.full(n, r_max, dtype=np.float64),
                np.linspace(0.0, 90.0, n))
    elif init == "diag45":
        # all on the 45-deg diagonal (NOT a mirror axis -> free to move in
        # angle), radii spread uniformly across the feasible band.
        return (np.linspace(r_min, r_max, n),
                np.full(n, 45.0, dtype=np.float64))
    elif init == "random":
        return None                                  # filled later
    else:
        base = _parse_positions(init)
    if len(base) < n:
        raise ValueError(f"init has {len(base)} < n={n} positions")
    base = base[:n]
    r = np.array([p[0] for p in base], dtype=np.float64)
    th = np.array([p[1] for p in base], dtype=np.float64)
    return r, th


class _Placement:
    """Learnable sensor positions under a chosen parameterization,
    with projection onto the feasible band [r_min, r_max] x [0, 90].

    Exposes the SAME interface regardless of parameterization:
      xy()        -> (n, 2) differentiable Cartesian points in [0, 1]
      project_()  -> clamp back into the feasible band (in place)
      rtheta()    -> (n, 2) numpy (r, theta_deg) for logging / plots
      params      -> list of leaf tensors to hand the optimizer
    """

    def __init__(self, r0, th0, param, r_min, r_max, device):
        self.param = param
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        if param == "cartesian":
            x = r0 * np.cos(np.deg2rad(th0))
            y = r0 * np.sin(np.deg2rad(th0))
            self.xy_par = torch.tensor(
                np.stack([x, y], axis=1), dtype=torch.float32,
                device=device, requires_grad=True)
            self.params = [self.xy_par]
        elif param in ("polar-deg", "polar-rad"):
            self.r_par = torch.tensor(r0, dtype=torch.float32,
                                      device=device, requires_grad=True)
            th_init = th0 if param == "polar-deg" else np.deg2rad(th0)
            self.th_par = torch.tensor(th_init, dtype=torch.float32,
                                       device=device, requires_grad=True)
            self.params = [self.r_par, self.th_par]
        else:
            raise ValueError(f"unknown param mode {param!r}")

    def xy(self):
        if self.param == "cartesian":
            return self.xy_par
        th_rad = (self.th_par * (np.pi / 180.0)
                  if self.param == "polar-deg" else self.th_par)
        x = self.r_par * torch.cos(th_rad)
        y = self.r_par * torch.sin(th_rad)
        return torch.stack([x, y], dim=1)

    @torch.no_grad()
    def project_(self):
        if self.param == "cartesian":
            x = self.xy_par[:, 0]
            y = self.xy_par[:, 1]
            r = torch.sqrt(x * x + y * y).clamp(self.r_min, self.r_max)
            th = torch.atan2(y, x).clamp(0.0, np.pi / 2.0)
            self.xy_par.copy_(torch.stack(
                [r * torch.cos(th), r * torch.sin(th)], dim=1))
        else:
            self.r_par.clamp_(self.r_min, self.r_max)
            hi = 90.0 if self.param == "polar-deg" else np.pi / 2.0
            self.th_par.clamp_(0.0, hi)

    def rtheta(self):
        with torch.no_grad():
            if self.param == "cartesian":
                x = self.xy_par[:, 0].cpu().numpy()
                y = self.xy_par[:, 1].cpu().numpy()
                return np.stack([np.hypot(x, y),
                                 np.degrees(np.arctan2(y, x))], axis=1)
            r = self.r_par.cpu().numpy()
            th = self.th_par.cpu().numpy()
            if self.param == "polar-rad":
                th = np.degrees(th)
            return np.stack([r, th], axis=1)


def _frame_indices(n_all, max_frames):
    if n_all <= max_frames:
        return np.arange(n_all)
    return np.linspace(0, n_all - 1, max_frames).astype(int)


def _xy_of(pos):
    """(., n, 2) or (n, 2) polar (r, theta_deg) -> Cartesian, same shape."""
    r = pos[..., 0]
    th = np.deg2rad(pos[..., 1])
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)


def _spread_random_init(rng, n, r_min, r_max, min_sep, tries=300):
    """Random (r, theta) init whose sensors are pairwise >= min_sep apart in
    Cartesian space (rejection sampling), so no restart starts collapsed.
    Returns the best-separated draw if none fully clears min_sep."""
    best = None
    for _ in range(tries):
        r = rng.uniform(r_min, r_max, n)
        th = rng.uniform(0.0, 90.0, n)
        xy = np.stack([r * np.cos(np.deg2rad(th)),
                       r * np.sin(np.deg2rad(th))], axis=1)
        d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
        d[np.eye(n, dtype=bool)] = np.inf
        m = float(d.min())
        if m >= min_sep:
            return r, th
        if best is None or m > best[0]:
            best = (m, r, th)
    return best[1], best[2]


def run(basis_path, *, traj_path=None, npz_dir=None, K=12, n=6,
        init="abcdef", param="cartesian", r_min=0.2, r_max=0.98,
        epochs=300, lr=1e-3, pos_lr=2e-2, val_frac=0.2, seed=7,
        nt=300, drop_first_steps=1, limit=400, random_subsample=False,
        channels=64, dilations=(1, 2, 4, 8, 16, 32, 64), kernel=3,
        device=None, verbose=True, min_sep=0.1, rep_coef=50.0) -> dict:
    """Load the data, then optimize once from `init` (see _optimize)."""
    Phi, a_np, (nx, ny) = _load_phi_and_a(
        basis_path, traj_path, npz_dir, K, nt, drop_first_steps,
        limit, random_subsample=random_subsample, seed=seed)
    return _optimize(Phi, a_np, nx, ny, n=n, init=init, param=param,
                     r_min=r_min, r_max=r_max, epochs=epochs, lr=lr,
                     pos_lr=pos_lr, val_frac=val_frac, seed=seed,
                     channels=channels, dilations=dilations, kernel=kernel,
                     device=device, verbose=verbose, min_sep=min_sep,
                     rep_coef=rep_coef)


def _optimize(Phi, a_np, nx, ny, *, n=6, init="random", init_positions=None,
              param="cartesian", r_min=0.2, r_max=0.98, epochs=300, lr=1e-3,
              pos_lr=2e-2, val_frac=0.2, seed=7, channels=64,
              dilations=(1, 2, 4, 8, 16, 32, 64), kernel=3, device=None,
              verbose=True, min_sep=0.1, rep_coef=50.0) -> dict:
    """Gradient-optimize sensor positions on ALREADY-loaded Phi/a. Split out of
    run() so a multi-start driver can load the data ONCE and optimize from many
    random inits cheaply."""
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    K = Phi.shape[1]
    n_sim, _, Nt = a_np.shape

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
    a_std = a_t.std(dim=(0, 2), keepdim=True).clamp_min(1e-8)
    a_norm = a_t / a_std

    if init_positions is not None:
        arr = np.asarray(init_positions, dtype=np.float64).reshape(-1, 2)
        r0, th0 = arr[:, 0].copy(), arr[:, 1].copy()
    else:
        ip = _init_positions(init, n, r_min, r_max)
        if ip is None:
            r0, th0 = _spread_random_init(rng, n, r_min, r_max, min_sep)
        else:
            r0, th0 = ip
    place = _Placement(r0, th0, param, r_min, r_max, dev)
    init_pos = np.stack([r0, th0], axis=1).copy()

    def measure(idx):
        """Differentiable sensor time series for sims `idx`: (B, n, Nt)."""
        xy = place.xy()                                    # (n, 2) in [0,1]
        grid = (2.0 * xy - 1.0)[None, None]                # (1,1,n,2)
        phi_at = F.grid_sample(Phi_img, grid, mode="bilinear",
                               padding_mode="border",
                               align_corners=True)          # (1,K,1,n)
        phi_at = phi_at[0, :, 0, :]                          # (K, n)
        return torch.einsum("kn,bkt->bnt", phi_at, a_t[idx])

    with torch.no_grad():
        y0 = measure(torch.tensor(tr_idx, device=dev))
        y_std = y0.std().clamp_min(1e-8)

    model = create_model("bitcn", n_in=n, n_out=K, channels=channels,
                         dilations=list(dilations), kernel=kernel,
                         dropout=0.0, causal=False).to(dev)
    opt = torch.optim.Adam(
        [{"params": model.parameters(), "lr": lr},
         {"params": place.params, "lr": pos_lr}])

    tr_i = torch.tensor(tr_idx, device=dev)
    vl_i = torch.tensor(val_idx, device=dev)
    pos_hist = [init_pos]
    val_hist = []
    init_xy = _xy_of(init_pos)                             # (n, 2)
    # FULL per-epoch position history (r, theta_deg): index 0 = init,
    # index ep+1 = after epoch ep. Drives the migration animation.
    pos_full = np.zeros((epochs + 1, n, 2), dtype=np.float64)
    pos_full[0] = init_pos
    move_hist = np.zeros((epochs, n), dtype=np.float64)
    best_val = np.inf
    best_pos = init_pos

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        y = measure(tr_i) / y_std
        loss = F.mse_loss(model(y), a_norm[tr_i])
        if rep_coef > 0.0 and min_sep > 0.0:
            # hinge repulsion: penalize only pairs closer than min_sep, so it
            # prevents sensors collapsing together without biasing spread
            # layouts (zero penalty once every pair is >= min_sep apart).
            xy = place.xy()                                # (n, 2)
            d = torch.cdist(xy, xy)                        # (n, n)
            iu = torch.triu_indices(n, n, offset=1, device=d.device)
            loss = loss + rep_coef * torch.relu(
                min_sep - d[iu[0], iu[1]]).pow(2).sum()
        loss.backward()
        opt.step()
        place.project_()
        cur = place.rtheta()                               # (n, 2)
        pos_full[ep + 1] = cur
        cxy = _xy_of(cur)
        move_hist[ep] = np.sqrt(((cxy - init_xy) ** 2).sum(axis=1))

        if (ep + 1) % max(1, epochs // 30) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                yv = measure(vl_i) / y_std
                vloss = float(F.mse_loss(model(yv), a_norm[vl_i]))
            pos_hist.append(cur.copy())
            val_hist.append((ep + 1, float(loss.detach()), vloss))
            if vloss < best_val:
                best_val, best_pos = vloss, cur.copy()
            if verbose:
                print(f"  ep {ep + 1:4d}  train {float(loss.detach()):.4e}  "
                      f"val {vloss:.4e}", flush=True)

    final_pos = place.rtheta()
    fxy, ixy = _xy_of(final_pos), _xy_of(init_pos)
    moved = np.sqrt(((fxy - ixy) ** 2).sum(axis=1))
    return dict(init_pos=init_pos, final_pos=final_pos,
                best_pos=best_pos, best_val=best_val,
                pos_hist=pos_hist, val_hist=val_hist, moved=moved,
                move_hist=move_hist, pos_full=pos_full, param=param,
                K=K, n=n, r_min=r_min, r_max=r_max, n_sim=n_sim, Nt=Nt)


def _render(res, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axm, axl) = plt.subplots(
        1, 3, figsize=(18, 5.6),
        gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.0]),
        constrained_layout=True)
    th = np.linspace(0, 90, 200)
    ax.plot(np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th)),
            color="0.35", lw=2)
    ax.plot([0, 1.05], [0, 0], color="0.7", lw=1)
    ax.plot([0, 0], [0, 1.05], color="0.7", lw=1)
    for rb in (res["r_min"], res["r_max"]):
        ax.plot(rb * np.cos(np.deg2rad(th)), rb * np.sin(np.deg2rad(th)),
                color="#2a9d8f", lw=1, ls="-", alpha=0.6)

    def xy(p):
        return (p[:, 0] * np.cos(np.deg2rad(p[:, 1])),
                p[:, 0] * np.sin(np.deg2rad(p[:, 1])))

    ix, iy = xy(res["init_pos"])
    fx, fy = xy(res["best_pos"])
    ph = res.get("pos_hist", [])
    if len(ph) > 1:
        paths = np.stack(list(ph), axis=0)
        for k in range(res["n"]):
            px = paths[:, k, 0] * np.cos(np.deg2rad(paths[:, k, 1]))
            py = paths[:, k, 0] * np.sin(np.deg2rad(paths[:, k, 1]))
            ax.plot(px, py, "-", color="0.55", lw=1.2, alpha=0.9, zorder=3)
    ax.scatter(ix, iy, s=140, marker="s", facecolor="none",
               edgecolor="0.35", linewidth=1.8, zorder=4, label="init")
    ax.scatter(fx, fy, s=70, marker="o", color="#e63946",
               edgecolor="black", linewidth=0.8, zorder=6, label="optimized")
    for k in range(res["n"]):
        ax.annotate(str(k + 1), (fx[k], fy[k]), xytext=(7, 5),
                    textcoords="offset points", fontsize=9,
                    fontweight="bold", color="#e63946", zorder=7)
    ax.set_xlim(-0.08, 1.15)
    ax.set_ylim(-0.08, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")
    ax.set_title(f"Placement paths (n={res['n']}, K={res['K']}, "
                 f"param={res.get('param', '?')})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    mh = res.get("move_hist")
    if mh is not None and mh.size:
        ep = np.arange(1, mh.shape[0] + 1)
        for k in range(res["n"]):
            axm.plot(ep, mh[:, k], lw=1.4, label=f"sensor {k + 1}")
        axm.set_xlabel("epoch")
        axm.set_ylabel("distance moved from init (canonical)")
        axm.set_title("sensor movement (flat = converged)")
        axm.legend(fontsize=8, ncol=2)
        axm.grid(alpha=0.3)

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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_location_anim(res, out_path, fps=12, max_frames=150):
    """Animate every sensor's position over the optimization epochs on
    the quarter disk, with a growing trail per sensor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    pf = np.asarray(res["pos_full"])                       # (E+1, n, 2)
    xy = _xy_of(pf)                                        # (E+1, n, 2)
    E1, nS, _ = xy.shape
    frames = _frame_indices(E1, max_frames)

    fig, ax = plt.subplots(figsize=(6.8, 6.8),
                           constrained_layout=True)
    th = np.linspace(0, 90, 200)
    ax.plot(np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th)),
            color="0.35", lw=2)
    ax.plot([0, 1.05], [0, 0], color="0.7", lw=1)
    ax.plot([0, 0], [0, 1.05], color="0.7", lw=1)
    for rb in (res["r_min"], res["r_max"]):
        ax.plot(rb * np.cos(np.deg2rad(th)), rb * np.sin(np.deg2rad(th)),
                color="#2a9d8f", lw=1, alpha=0.6)
    colors = plt.cm.tab10(np.arange(nS) % 10)
    ax.scatter(xy[0, :, 0], xy[0, :, 1], s=130, marker="s",
               facecolor="none", edgecolor="0.4", linewidth=1.6,
               zorder=3, label="init")
    trails = [ax.plot([], [], "-", color=colors[k], lw=1.4,
                      alpha=0.75, zorder=4)[0] for k in range(nS)]
    dots = ax.scatter(xy[0, :, 0], xy[0, :, 1], s=80, c=colors,
                      edgecolor="black", linewidth=0.7, zorder=6)
    txt = ax.text(0.02, 1.10, "", fontsize=13, fontweight="bold",
                  color="#1d3557")
    ax.set_xlim(-0.08, 1.15)
    ax.set_ylim(-0.08, 1.18)
    ax.set_aspect("equal")
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")
    ax.set_title(f"Sensor migration (n={nS}, K={res['K']}, "
                 f"param={res.get('param', '?')})")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    def update(fi):
        e = int(frames[fi])
        for k in range(nS):
            trails[k].set_data(xy[:e + 1, k, 0], xy[:e + 1, k, 1])
        dots.set_offsets(xy[e, :, :])
        txt.set_text(f"epoch {e}/{E1 - 1}")
        return trails + [dots, txt]

    print(f"rendering {len(frames)} migration frames at {fps} fps "
          f"-> {out_path}", flush=True)
    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // fps, blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return Path(out_path)


def _save_history(res, path):
    """Persist everything the two renderers need, so the figures /
    animation can be redrawn later without re-optimizing."""
    ph = res.get("pos_hist", [])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pos_full=np.asarray(res["pos_full"]),
        pos_hist=(np.stack(list(ph), axis=0) if len(ph)
                  else np.zeros((0, res["n"], 2))),
        move_hist=np.asarray(res["move_hist"]),
        val_hist=(np.asarray(res["val_hist"], dtype=np.float64)
                  if res["val_hist"] else np.zeros((0, 3))),
        init_pos=res["init_pos"], best_pos=res["best_pos"],
        final_pos=res["final_pos"], moved=res["moved"],
        K=np.int64(res["K"]), n=np.int64(res["n"]),
        r_min=np.float64(res["r_min"]), r_max=np.float64(res["r_max"]),
        param=np.str_(res.get("param", "")))


def _load_history(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(
            pos_full=z["pos_full"],
            pos_hist=[z["pos_hist"][i] for i in range(len(z["pos_hist"]))],
            move_hist=z["move_hist"],
            val_hist=[tuple(row) for row in z["val_hist"]],
            init_pos=z["init_pos"], best_pos=z["best_pos"],
            final_pos=z["final_pos"], moved=z["moved"],
            K=int(z["K"]), n=int(z["n"]),
            r_min=float(z["r_min"]), r_max=float(z["r_max"]),
            param=str(z["param"]))


def _print_summary(res):
    print(f"\nfinal positions (param={res.get('param')}, "
          f"best val={res['best_val']:.4e}):")
    print(f"  {'#':>2}  {'r_init':>7} {'th_init':>7}  ->  "
          f"{'r_opt':>7} {'th_opt':>7}  {'d_r':>7} {'d_th':>7} "
          f"{'moved':>7}")
    ip_all, fp_all = res["init_pos"], res["best_pos"]
    for i in range(res["n"]):
        ip, fp = ip_all[i], fp_all[i]
        print(f"  {i + 1:>2}  {ip[0]:7.3f} {ip[1]:7.1f}  ->  "
              f"{fp[0]:7.3f} {fp[1]:7.1f}  {fp[0] - ip[0]:+7.3f} "
              f"{fp[1] - ip[1]:+7.1f} {res['moved'][i]:7.3f}")
    d_r = np.abs(fp_all[:, 0] - ip_all[:, 0])
    d_th = np.abs(fp_all[:, 1] - ip_all[:, 1])
    print(f"\n  mean |delta r| = {d_r.mean():.4f}   "
          f"mean |delta theta| = {d_th.mean():.2f} deg")
    pos_json = json.dumps([[round(float(r), 4), round(float(t), 2)]
                           for r, t in res["best_pos"]])
    print(f"\noptimized positions JSON:\n  {pos_json}")
    tot = float(res["moved"].sum())
    print(f"\ntotal movement (canonical units): {tot:.3f}")
    if tot < 0.02 * res["n"]:
        print("  -> positions barely moved: the init layout is at "
              "(or near) a local optimum for this objective.")
    else:
        print("  -> positions moved substantially: the init layout is "
              "NOT optimal; the paths show the improving direction.")
    return pos_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basis", default=None)
    ap.add_argument("--traj", default=None)
    ap.add_argument(
        "--npz-dir",
        default="/data/3D_wafer_bonding/sim_dataset_big_firehorse_1_and_2/",
        help="dataset to project onto the basis (used unless --traj is "
        "given); defaults to the firehorse_1_and_2 merged set")
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--init", default="abcdef",
                    help="'abcdef'; 'uniform-outer' (n sensors on the outer "
                    "ring at angles linspace(0,90,n)); 'diag45' (n sensors on "
                    "the 45-deg ray, radii linspace(r_min,r_max,n)); 'random'; "
                    "or positions as inline JSON [[r,theta],...] OR a path to a "
                    "JSON file. Prefer a FILE for custom positions -- shells "
                    "(zsh) mangle inline [ ] brackets.")
    ap.add_argument("--param", default="cartesian",
                    choices=["cartesian", "polar-rad", "polar-deg"],
                    help="position parameterization. 'cartesian' "
                    "(default) makes radial + tangential exploration "
                    "physically isotropic; 'polar-deg' is the old "
                    "throttled behavior (theta in degrees moves ~60x "
                    "slower tangentially). Use both to A/B test whether "
                    "angle genuinely matters.")
    ap.add_argument("--r-min", type=float, default=0.2)
    ap.add_argument("--r-max", type=float, default=0.98)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-lr", type=float, default=2e-2,
                    help="learning rate for sensor positions "
                    "(default 2e-2)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-sep", type=float, default=0.1,
                    help="minimum pairwise sensor spacing (normalized units); "
                    "a hinge repulsion penalty keeps sensors from collapsing "
                    "together. 0 disables it.")
    ap.add_argument("--rep-coef", type=float, default=50.0,
                    help="weight of the min-sep repulsion penalty (default 50)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--drop-first-steps", type=int, default=1)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--random-subsample", action="store_true",
                    help="pick the --limit sims at RANDOM (via "
                    "symlinks) instead of the loader's first-N.")
    ap.add_argument("--out", default=None,
                    help="3-panel static figure (paths + movement + loss)")
    ap.add_argument("--anim-out", default=None,
                    help="GIF animating sensor migration over epochs")
    ap.add_argument("--anim-fps", type=int, default=12)
    ap.add_argument("--anim-max-frames", type=int, default=150)
    ap.add_argument("--save-history", default=None,
                    help="npz of the full per-epoch position history + "
                    "curves, so figures can be redrawn without retraining")
    ap.add_argument("--from-history", default=None,
                    help="skip training; redraw --out / --anim-out from "
                    "a --save-history npz")
    ap.add_argument("--positions-json", default=None)
    args = ap.parse_args()

    if args.from_history:
        res = _load_history(args.from_history)
        res["best_val"] = float("nan")
        pos_json = _print_summary(res)
        if args.positions_json:
            Path(args.positions_json).write_text(pos_json)
            print(f"wrote {args.positions_json}")
        if args.out:
            _render(res, Path(args.out))
            print(f"wrote {args.out}")
        if args.anim_out:
            _render_location_anim(res, Path(args.anim_out),
                                  fps=args.anim_fps,
                                  max_frames=args.anim_max_frames)
            print(f"wrote {args.anim_out}")
        return 0

    if not args.basis or not Path(args.basis).is_file():
        print(f"basis not found: {args.basis}", file=sys.stderr)
        return 2
    res = run(args.basis, traj_path=args.traj, npz_dir=args.npz_dir,
              K=args.K, n=args.n, init=args.init, param=args.param,
              r_min=args.r_min, r_max=args.r_max, epochs=args.epochs,
              lr=args.lr, pos_lr=args.pos_lr, val_frac=args.val_frac,
              seed=args.seed, nt=args.nt,
              drop_first_steps=args.drop_first_steps, limit=args.limit,
              random_subsample=args.random_subsample,
              min_sep=args.min_sep, rep_coef=args.rep_coef)

    pos_json = _print_summary(res)
    if args.positions_json:
        Path(args.positions_json).write_text(pos_json)
        print(f"wrote {args.positions_json}")
    if args.save_history:
        _save_history(res, Path(args.save_history))
        print(f"wrote {args.save_history}")
    if args.out:
        _render(res, Path(args.out))
        print(f"wrote {args.out}")
    if args.anim_out:
        _render_location_anim(res, Path(args.anim_out),
                              fps=args.anim_fps,
                              max_frames=args.anim_max_frames)
        print(f"wrote {args.anim_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
