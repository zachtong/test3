"""Build a tiny converted-NPZ fixture matching the 3D schema.

Used by the loader smoke test (and as a reference for what a real converted
NPZ should look like). Variable T_i and N_upper_i / N_lower_i across steps
exercise the adaptive-mesh path; native (x, y) stay in the first quadrant
so the quarter-symmetry validation in the loader does not trip.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np


def make_mock_3d_npz(out_path: Path,
                     n_steps: int = 2,
                     t_per_step: tuple[int, ...] = (5, 4),
                     n_upper_per_step: tuple[int, ...] = (50, 60),
                     n_lower_per_step: tuple[int, ...] = (40, 55),
                     R: float = 0.15,
                     seed: int = 0) -> None:
    """Synthesize one converted-NPZ fixture; coordinates are quarter-disk.

    Field model: a separable axisymmetric blob `u_z(x, y, t) = sin(pi r)
    * sin(2 pi t)` plus a small C4-symmetric non-axisymmetric ripple
    `0.1 cos(4 theta) * sin(pi r) * t` so POD has more than one mode.
    Coordinates are randomly sampled in the first quadrant of the unit disk
    so that interpolation onto a regular grid is a non-trivial operation.
    """
    assert len(t_per_step) == n_steps
    assert len(n_upper_per_step) == n_steps
    assert len(n_lower_per_step) == n_steps
    rng = np.random.default_rng(seed)
    S = int(sum(t_per_step))

    payload: dict[str, np.ndarray] = {}
    # Global sample-index arrays.
    sample_step_index = np.empty(S, dtype=np.int64)
    sample_time_index = np.empty(S, dtype=np.int64)
    sample_tReal = np.empty(S, dtype=np.float64)
    sample_bonding_front = np.empty(S, dtype=np.float32)

    t_offset = 0.0
    k = 0
    for i in range(n_steps):
        Ti = t_per_step[i]
        N_up = n_upper_per_step[i]
        N_lo = n_lower_per_step[i]

        # Quarter-disk uniform-ish sample (rejection on unit disk).
        def quarter_disk_points(N):
            pts = []
            while len(pts) < N:
                cand = rng.random((N, 2)) * R              # in [0, R]
                in_disk = (cand[:, 0] ** 2 + cand[:, 1] ** 2) <= R * R
                pts.extend(cand[in_disk].tolist())
            arr = np.asarray(pts[:N], dtype=np.float32)
            return arr

        xy_up = quarter_disk_points(N_up)
        xy_lo = quarter_disk_points(N_lo)
        # z dummy (lower at 0, upper at slight separation -- typical).
        z_up = np.full(N_up, 1e-3, dtype=np.float32)
        z_lo = np.zeros(N_lo, dtype=np.float32)
        coords_up = np.stack(
            [xy_up[:, 0], xy_up[:, 1], z_up], axis=0)             # (3, N_up)
        coords_lo = np.stack(
            [xy_lo[:, 0], xy_lo[:, 1], z_lo], axis=0)             # (3, N_lo)

        # Time grid within this step.
        t_local = t_offset + np.linspace(0.0, 1.0, Ti, dtype=np.float64)

        # Build the synthetic field at each native point and time.
        r_up = np.sqrt(xy_up[:, 0] ** 2 + xy_up[:, 1] ** 2) / R    # (N_up,) in [0, 1]
        theta_up = np.arctan2(xy_up[:, 1], xy_up[:, 0])            # (N_up,) in [0, pi/2]
        base_up = np.sin(np.pi * r_up)                              # (N_up,)
        ripple_up = 0.1 * np.cos(4.0 * theta_up) * base_up          # (N_up,)
        disp_up = (base_up[None, :] * np.sin(2 * np.pi * t_local)[:, None]
                   + ripple_up[None, :] * t_local[:, None])         # (Ti, N_up)
        disp_up = disp_up.astype(np.float32)

        r_lo = np.sqrt(xy_lo[:, 0] ** 2 + xy_lo[:, 1] ** 2) / R
        disp_lo = (np.sin(np.pi * r_lo)[None, :]
                   * (0.1 * np.sin(2 * np.pi * t_local))[:, None]
                   ).astype(np.float32)

        thickness_up = np.full((Ti, N_up), 5e-4, dtype=np.float32)
        thickness_lo = np.full((Ti, N_lo), 5e-4, dtype=np.float32)
        bonding_front = (0.2 + 0.8 * t_local).astype(np.float32)    # rising

        prefix = f"step_{i:04d}"
        payload[f"{prefix}_coordinates_lower"] = coords_lo
        payload[f"{prefix}_coordinates_upper"] = coords_up
        payload[f"{prefix}_displacement_z_corrected_lower"] = disp_lo
        payload[f"{prefix}_displacement_z_corrected_upper"] = disp_up
        payload[f"{prefix}_thickness_lower"] = thickness_lo
        payload[f"{prefix}_thickness_upper"] = thickness_up
        payload[f"{prefix}_bonding_front"] = bonding_front
        payload[f"{prefix}_tReal"] = t_local
        payload[f"{prefix}_num_time_points"] = np.int64(Ti)
        payload[f"{prefix}_num_points"] = np.int64(N_up + N_lo)
        payload[f"{prefix}_num_lower_points"] = np.int64(N_lo)
        payload[f"{prefix}_num_upper_points"] = np.int64(N_up)
        payload[f"{prefix}_z_min"] = np.float64(0.0)
        payload[f"{prefix}_z_max"] = np.float64(1e-3)

        # Append to sample-index arrays. Make tReal monotonic across steps
        # with one duplicate at the boundary to exercise the dedupe path.
        for j in range(Ti):
            sample_step_index[k] = i
            sample_time_index[k] = j
            sample_tReal[k] = t_local[j]
            sample_bonding_front[k] = bonding_front[j]
            k += 1
        # Drift the next step's start so step boundaries can repeat exactly.
        t_offset = float(t_local[-1])

    # Global metadata + sample-flat duplicates.
    payload["sample_step_index"] = sample_step_index
    payload["sample_time_index_within_step"] = sample_time_index
    payload["sample_tReal"] = sample_tReal
    payload["sample_bonding_front"] = sample_bonding_front
    payload["sample_num_time_points_in_step"] = np.array(
        [t_per_step[i] for i in sample_step_index], dtype=np.int64)
    payload["sample_num_points"] = np.array(
        [n_upper_per_step[i] + n_lower_per_step[i]
         for i in sample_step_index], dtype=np.int64)
    payload["sample_num_lower_points"] = np.array(
        [n_lower_per_step[i] for i in sample_step_index], dtype=np.int64)
    payload["sample_num_upper_points"] = np.array(
        [n_upper_per_step[i] for i in sample_step_index], dtype=np.int64)
    payload["sample_z_min"] = np.zeros(S, dtype=np.float64)
    payload["sample_z_max"] = np.full(S, 1e-3, dtype=np.float64)

    payload["num_samples"] = np.int64(S)
    payload["num_wafer_steps"] = np.int64(n_steps)
    # Mirror the real converter: claim the source JSON had one more step
    # than what was converted (trim-last-step invariant).
    payload["num_original_wafer_steps"] = np.int64(n_steps + 1)
    payload["last_step_removed"] = np.bool_(True)
    payload["num_valid_wafer_steps"] = np.int64(n_steps)
    payload["skipped_step_count"] = np.int64(0)
    payload["step_metadata_json"] = np.array("[]", dtype=object)
    payload["skipped_steps_json"] = np.array("[]", dtype=object)
    payload["source_json"] = np.array(str(out_path), dtype=object)
    payload["source_json_name"] = np.array(out_path.name, dtype=object)
    payload["json_file_size_bytes"] = np.int64(0)
    payload["converter_version"] = np.array("mock", dtype=object)
    payload["minimal_fields"] = np.bool_(True)
    payload["repaired_or_not"] = np.bool_(False)
    payload["invalid_json_policy"] = np.array("skip_no_repair", dtype=object)
    payload["complete_json_required"] = np.bool_(True)
    payload["coordinate_system"] = np.array("cartesian_3d", dtype=object)
    payload["coordinate_layout"] = np.array("(3,N)", dtype=object)
    payload["wafer_split_mode"] = np.array("z_min_z_max", dtype=object)
    payload["z_correction_mode"] = np.array(
        "shell_umz_plus_half_thickness_arz", dtype=object)
    payload["z_correction_formula"] = np.array(
        "displacement_z_corrected = shell.umz + 0.5 * shell.d * arz",
        dtype=object)
    payload["array_float_dtype"] = np.array("float32", dtype=object)
    payload["time_dtype"] = np.array("float64", dtype=object)

    # Optional COMSOL physical metadata.
    payload["contactTime"] = np.float64(0.05)
    payload["releaseTime_LW"] = np.float64(0.0)
    payload["releaseTime_UW"] = np.float64(0.0)
    payload["hGap"] = np.float64(1e-4)
    payload["modelName"] = np.array("mock_model", dtype=object)
    payload["allParams_json"] = np.array("{}", dtype=object)
    payload["expr"] = np.array(
        ["shell.umx", "shell.umy", "shell.umz",
         "arx", "ary", "arz", "shell.d"], dtype=object)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)


def make_bad_3d_npz(out_path,
                    mode: str = "nonzero_skipped",
                    **mock_kwargs) -> None:
    """Build a converted-NPZ fixture that should FAIL `preflight_npz`.

    Mirrors the kinds of broken-but-zip-valid NPZs the real converter is
    known to emit (per the converter author's note: some files have
    internal step skipped because of d3/d6/d2 vs coordinate-count
    mismatches). Used by the skip-tolerance tests.

    Modes (each sabotages exactly one invariant):
      'nonzero_skipped'    -- skipped_step_count = 1
      'missing_step'       -- drop every step_0001_* key (prefix gap)
      'wrong_disp_shape'   -- displacement_z_corrected_upper has wrong shape
      'bad_quadrant'       -- step_0000_coordinates_upper has a negative x
      'sample_shape_off'   -- sample_tReal length != num_samples
      'last_step_kept'     -- last_step_removed = False
      'treal_huge_back'    -- inject a backward jump in sample_tReal that
                              exceeds _TREAL_BACKWARD_FACTOR * typ_dt
    """
    make_mock_3d_npz(out_path, **mock_kwargs)
    with np.load(out_path, allow_pickle=True) as z:
        payload = {k: z[k] for k in z.files}

    if mode == "nonzero_skipped":
        payload["skipped_step_count"] = np.int64(1)
    elif mode == "missing_step":
        # require at least 2 steps to drop the second
        keys_to_drop = [k for k in payload if k.startswith("step_0001_")]
        if not keys_to_drop:
            raise ValueError("missing_step needs n_steps >= 2 in mock")
        for k in keys_to_drop:
            payload.pop(k)
        # but leave num_wafer_steps unchanged: now prefix count < nws
    elif mode == "wrong_disp_shape":
        # Drop one column so the array doesn't match num_upper_points.
        bad = payload["step_0000_displacement_z_corrected_upper"]
        payload["step_0000_displacement_z_corrected_upper"] = bad[:, :-1]
    elif mode == "bad_quadrant":
        payload["step_0000_coordinates_upper"][0, 0] = -0.5
    elif mode == "sample_shape_off":
        # Shrink sample_tReal to length S - 1; sample_step_index stays right.
        payload["sample_tReal"] = payload["sample_tReal"][:-1]
    elif mode == "last_step_kept":
        payload["last_step_removed"] = np.bool_(False)
    elif mode == "treal_huge_back":
        # Take the last sample and shove its time backward by 50x the
        # typical dt -- well past the relaxed factor of 10x. tReal is
        # otherwise monotonic in the mock so the median forward dt is
        # well-defined.
        treal = payload["sample_tReal"].copy()
        dt = np.diff(treal)
        typ_dt = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.01
        treal[-1] = treal[-2] - 50.0 * typ_dt
        payload["sample_tReal"] = treal
    else:
        raise ValueError(f"unknown sabotage mode {mode!r}")

    np.savez(out_path, **payload)
