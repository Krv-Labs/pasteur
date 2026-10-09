# pasteur-model

In-process ONNX Runtime scoring for Pasteur evaluations. `OnnxModel` implements `pasteur_core::Model`.

```rust
use pasteur_model::OnnxModel;

let model = OnnxModel::from_file(
    "model.onnx".as_ref(),
    "input",                 // ONNX input name
    feature_order,           // Vec<String>, the column order fed to the model
    Some(1),                 // positive class index: binary only (None → 1); None for other tasks
    None,                    // null fill: None → metadata.json absent_sentinel → NaN
    None,                    // contract path: None → {model_dir}/metadata.json
)?;
```

## Model contract (`metadata.json`)

This is an optional sidecar next to the `.onnx` file. It is the only thing that can verify **column order**, because the ONNX graph only knows the input width.

```json
{ "input": { "feature_order": ["TSH", "T3", "T4"], "absent_sentinel": 0.0 } }
```

- A width or order mismatch is an error, not a silent mis-score.
- A NaN probability output is an error. It means the model is not NaN-native, so pass `null_fill` or declare `absent_sentinel`.

## Build note

The `ort` dependency uses `download-binaries`, so the **first build downloads ONNX Runtime**. Scoring runs fully offline.
