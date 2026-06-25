"""COMSOL JSON -> compact 3D NPZ converter.

STUB. The 3D simulation export format is not finalized; this file is a
contract scaffold so calling code (loader, batch runner, tests) can be
written against the expected interface.

Once the JSON schema is fixed, this module should expose:

    parse(json_path: Path) -> dict     # raw arrays from one sim
    convert(json_path: Path, out_path: Path) -> None    # write compact NPZ

The NPZ written to disk is expected to carry (TENTATIVE field table -- adjust
once the schema is locked):

    field            shape           notes
    -----            -----           -----
    f                (Nx, Ny, Nt)    gap field (m) on the native grid
    x                (Nx,)           x coord, normalized by R into [-1, 1]
    y                (Ny,)           y coord, normalized by R into [-1, 1]
    tReal            (Nt,)           real experimental time (s), raw
    r_max            scalar          physical max in-disk r (m), provenance
    R                scalar          wafer radius (m), provenance
    source_json      scalar          source JSON path (provenance)

If the native simulation exports on a curvilinear / unstructured mesh rather
than a structured (Nx, Ny) grid, this contract changes -- the converter
would either (a) resample the export onto a structured (Nx, Ny) at convert
time, or (b) write the raw mesh + values and let the loader resample. Pick
once we see the actual JSON.
"""

from __future__ import annotations

from pathlib import Path


def parse(json_path: Path) -> dict:
    raise NotImplementedError(
        "data/json_to_npz_converter.py::parse: 3D JSON schema not yet "
        "defined. Mirror the 2D converter's shape (data/json_to_npz_"
        "converter.py in wafer_bonding_sparse_recon) once the COMSOL 3D "
        "export keys are known.")


def convert(json_path: Path, out_path: Path) -> None:
    raise NotImplementedError(
        "data/json_to_npz_converter.py::convert: 3D NPZ schema not yet "
        "defined. See module docstring for the tentative field table.")
