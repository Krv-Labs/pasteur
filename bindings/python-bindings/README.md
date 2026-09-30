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

`BlackoutSimulator` also accepts `window_frac`, `patient_id_col`, and `time_col` for longitudinal data. `JitterSimulator`'s `scale` is in units of the feature's standard deviation.

- Runs entirely in-process on your machine: no network calls, no telemetry.
- Takes and returns polars DataFrames. From pandas, use `pl.from_pandas(df)`. Call `fit` before `transform`.
- Covers the blackout and jitter simulators. Model scoring (`evaluate`, `compare`) and flipper live in the [`pasteur-cli`](https://github.com/Krv-Labs/pasteur/tree/main/cli) command-line tool.
- Output frames keep every input column, including patient identifiers. Handle them like the input.

Requires CPython ≥ 3.12. Wheels are abi3. This package is `pypasteur`; the `pasteur` package on PyPI is an unrelated project.

Documentation, including [security and data handling](https://docs.krv.ai/pasteur/security.html): [docs.krv.ai/pasteur](https://docs.krv.ai/pasteur/).

## Development

```bash
pip install maturin polars
maturin develop            # from this directory
python tests/smoke.py
```
