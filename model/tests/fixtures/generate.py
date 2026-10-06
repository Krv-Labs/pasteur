"""Regenerate the synthetic ONNX fixtures for model/tests/onnx_outputs.rs.

Every model is fit on random synthetic data (no patient data), so the files
only pin down *output formats*: one per shape sklearn-onnx emits for each
task. `expected.json` records onnxruntime's probabilities for PROBE rows so
the Rust tests can check column order, not just shape. Probabilities are
stored row-major ([row][class]) for tensors, and as per-row dicts for ZipMap.
Regressors record their one predicted value per row instead.

    pip install scikit-learn skl2onnx onnxruntime
    python model/tests/fixtures/generate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier

HERE = Path(__file__).parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2)).astype(np.float32)
PROBE = np.array([[0.0, 0.0], [2.0, -1.0], [-1.5, 2.0]], dtype=np.float32)

y_binary = (X[:, 0] + X[:, 1] > 0).astype(int)
y_multiclass = np.digitize(X[:, 0], [-0.5, 0.5])  # 0, 1, 2
y_multilabel = np.stack([X[:, 0] > 0, X[:, 1] > 0, X[:, 0] + X[:, 1] > 1], axis=1).astype(int)
names = np.array(["low", "mid", "high"])
# Drawn after every classifier target, so adding it left their fixtures as
# they were.
y_regression = (5.5 + X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=200)).astype(
    np.float32
)

fixtures = {
    # Binary, ZipMap seq(map(int64, float)) — the pre-existing default path.
    "binary_zipmap": (LogisticRegression().fit(X, y_binary), {}),
    # Multiclass, plain [n, 3] tensor.
    "multiclass_tensor": (LogisticRegression().fit(X, y_multiclass), {"zipmap": False}),
    # Multiclass, ZipMap with int64 keys.
    "multiclass_zipmap_int": (LogisticRegression().fit(X, y_multiclass), {}),
    # Multiclass, ZipMap with string keys (sorted: high, low, mid).
    "multiclass_zipmap_str": (LogisticRegression().fit(X, names[y_multiclass]), {}),
    # Multilabel, MultiOutputClassifier: seq(tensor [n, 2]) per label.
    "multilabel_multioutput": (
        MultiOutputClassifier(LogisticRegression()).fit(X, y_multilabel),
        {"zipmap": False},
    ),
    # Multilabel, native sigmoid head: plain [n, 3] tensor.
    "multilabel_tensor": (
        MLPClassifier(hidden_layer_sizes=(4,), max_iter=500, random_state=0).fit(X, y_multilabel),
        {"zipmap": False},
    ),
    # Regression: one [n, 1] float tensor named `variable`.
    "regression_linear": (LinearRegression().fit(X, y_regression), {}),
    "regression_forest": (
        RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0).fit(X, y_regression),
        {},
    ),
}

expected = {}
for name, (model, options) in fixtures.items():
    onx = to_onnx(
        model,
        X[:1],
        options={id(model): options} if options else None,
        target_opset=17,
        final_types=None,
    )
    # A stable input name, as Pasteur's --input-name default expects.
    onx.graph.input[0].name = "input"
    for node in onx.graph.node:
        node.input[:] = ["input" if i == "X" else i for i in node.input]
    path = OUT / f"{name}.onnx"
    path.write_bytes(onx.SerializeToString())

    sess = ort.InferenceSession(str(path))
    outputs = sess.run(None, {"input": PROBE})
    by_name = dict(zip([o.name for o in sess.get_outputs()], outputs))
    if "variable" in by_name:
        expected[name] = {
            "outputs": [o.name for o in sess.get_outputs()],
            "predictions": np.asarray(by_name["variable"]).reshape(-1).tolist(),
        }
        continue
    probs = by_name.get("output_probability", by_name.get("probabilities"))
    if isinstance(probs, list) and probs and isinstance(probs[0], dict):
        probs = [{str(k): float(v) for k, v in row.items()} for row in probs]
    elif isinstance(probs, list):
        # One [n, 2] tensor per label: column 1 is that label's positive.
        probs = np.stack([np.asarray(t)[:, 1] for t in probs], axis=1).tolist()
    else:
        probs = np.asarray(probs).tolist()
    expected[name] = {
        "outputs": [o.name for o in sess.get_outputs()],
        "probabilities": probs,
    }

expected["probe"] = PROBE.tolist()
(OUT / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
print("wrote", ", ".join(fixtures))
