# pypasteur

Python bindings for [pasteur-core](https://github.com/Krv-Labs/pasteur): data simulators for stress-testing clinical AI models, built on polars.

```bash
pip install pypasteur
```

```python
import polars as pl
import pypasteur

df = pl.read_parquet("clean.parquet")

# Null out `tsh` (and its companion columns) for a fraction of rows.
blackout = pypasteur.BlackoutSimulator("tsh", rate=0.1, random_state=42, companions=["tsh_measured"])
blackout.fit(df)
shifted = blackout.transform(df)

# Add scaled noise to `tsh`.
jitter = pypasteur.JitterSimulator("tsh", scale=1.0, random_state=42)
jitter.fit(df)
noisy = jitter.transform(df)
```

`BlackoutSimulator` also accepts `window_frac`, `patient_id_col`, and `time_col` for longitudinal data.

Requires CPython ≥ 3.12. Wheels are abi3.

## Development

```bash
pip install maturin polars
maturin develop            # from this directory
python tests/smoke.py
```
