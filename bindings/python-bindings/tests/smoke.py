"""Smoke test for an installed pypasteur wheel. Run: python tests/smoke.py"""

import importlib.metadata

import polars as pl

import pypasteur

assert pypasteur.__version__ == importlib.metadata.version("pypasteur"), pypasteur.__version__

N = 20
df = pl.DataFrame({"tsh": [float(i) for i in range(N)], "age": [40.0 + i for i in range(N)]})

blackout = pypasteur.BlackoutSimulator("tsh", rate=0.5, random_state=42)
blackout.fit(df)
out = blackout.transform(df)
assert out.shape == df.shape, out.shape
nulls = out["tsh"].null_count()
assert 0 < nulls < N, f"expected some but not all tsh nulled, got {nulls}"
assert out["age"].null_count() == 0, "blackout touched a column it shouldn't"

jitter = pypasteur.JitterSimulator("tsh", scale=1.0, random_state=42)
jitter.fit(df)
out = jitter.transform(df)
assert out.shape == df.shape, out.shape
assert not out["tsh"].equals(df["tsh"]), "jitter left tsh unchanged"
assert out["age"].equals(df["age"]), "jitter touched a column it shouldn't"

unfitted = pypasteur.JitterSimulator("tsh", scale=1.0, random_state=42)
try:
    unfitted.transform(df)
except ValueError as e:
    assert "fit before transform" in str(e), e
else:
    raise AssertionError("transform before fit should raise")

print("pypasteur smoke test OK")
