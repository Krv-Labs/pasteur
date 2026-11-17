# @krv-labs/pasteur

`pasteur-cli`: simulate data shifts, score ONNX models, compare them, and
render dataset cards. Prebuilt binaries for macOS (arm64, x64), Linux glibc
(x64, arm64), and Windows x64.

## Use

```bash
npx -y @krv-labs/pasteur --help
npx -y @krv-labs/pasteur simulate --input clean.parquet --output sim --feature f1
npm install -g @krv-labs/pasteur   # installs `pasteur-cli` (and the `pasteur` alias)
```

## Network-free

`pasteur-cli` does local compute only and never touches the network. Fetch
models and datasets from the Hugging Face Hub with the official `hf` CLI first,
then pass local paths:

```bash
hf download <org>/<model> --local-dir model
pasteur-cli evaluate flipper --sim-root sim --labels groups.parquet --model model/model.onnx
```

The binary ships in a per-platform package (`@krv-labs/pasteur-<platform>`)
installed as an optional dependency; nothing is downloaded at install or run
time. Set `PASTEUR_BINARY` to use a local build instead.

Source and docs: https://github.com/Krv-Labs/pasteur
