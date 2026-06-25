# Usage

> Skeleton. Most steps below are TODO until the NPZ schema and `data/loader.py`
> are filled in.

## 1. Convert simulation output to NPZ (TODO -- pending schema)

`data/json_to_npz_converter.py` is a stub. Once the 3D simulation export
format is finalized, this should parse each sim into a compact NPZ with the
fields tabulated in that file's docstring.

## 2. Load and check (TODO)

```bash
# stub example -- raises NotImplementedError today
python -c "from data.loader import load_dataset; r,s=load_dataset('/path', nx=128, ny=128, nt=300, limit=5)"
```

## 3. Train

```bash
python scripts/train.py --config configs/default.yaml --data.npz_dir /path/to/3d_npz
```

`train.py` will run once `data/loader.py` is implemented and a real config is
written. The training loop, normalization, loss, BiTCN architecture, and
checkpoint paths are all ported and ready.

## 4. Inspect results

```bash
python scripts/visualize.py outputs/<tag>/results.json
```

This works as-is; the `ResultSet` JSON schema is identical to the 2D codebase.
