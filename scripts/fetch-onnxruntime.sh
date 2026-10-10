#!/usr/bin/env bash
# Download Microsoft's ONNX Runtime release for this host, verify it, and print
# the path of its shared library. pasteur-cli loads that library at run time:
#
#   export ORT_DYLIB_PATH="$(scripts/fetch-onnxruntime.sh)"
#   cargo test --workspace --locked --exclude pypasteur-bindings
#
# The official builds need only glibc 2.27 (Ubuntu 18.04+, Debian 10+, RHEL 8+).
# ORT_VERSION and ORT_CACHE override the pinned version and the download dir;
# a version without a pinned digest is refused unless ORT_SHA256 is given.
set -euo pipefail

version="${ORT_VERSION:-1.28.0}"
cache="${ORT_CACHE:-$(cd "$(dirname "$0")/.." && pwd)/target/onnxruntime}"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)         asset="onnxruntime-linux-x64-${version}.tgz";     lib="libonnxruntime.so" ;;
  Linux-aarch64)        asset="onnxruntime-linux-aarch64-${version}.tgz"; lib="libonnxruntime.so" ;;
  Darwin-arm64)         asset="onnxruntime-osx-arm64-${version}.tgz";     lib="libonnxruntime.dylib" ;;
  *) echo "no Microsoft ONNX Runtime build for $(uname -s)-$(uname -m); set ORT_DYLIB_PATH yourself" >&2; exit 1 ;;
esac

# sha256 of the release assets at https://github.com/microsoft/onnxruntime/releases
case "${asset}" in
  onnxruntime-linux-x64-1.28.0.tgz)     pinned=a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407 ;;
  onnxruntime-linux-aarch64-1.28.0.tgz) pinned=e15ff8b5d85afe6c144d97c6fd432254bf76a219daaf17658087d6ecb3e8f0bb ;;
  onnxruntime-osx-arm64-1.28.0.tgz)     pinned=1268b359718099bde2cedb55787f182a130067bc4f31e8c88478c445b850d3d8 ;;
  *) pinned="" ;;
esac
want="${ORT_SHA256:-${pinned}}"
if [ -z "${want}" ]; then
  echo "no pinned sha256 for ${asset}; set ORT_SHA256 to verify it" >&2
  exit 1
fi

dir="${cache}/${asset%.tgz}"
if [ ! -e "${dir}/lib/${lib}" ]; then
  mkdir -p "${cache}"
  tmp="$(mktemp "${cache}/.${asset}.XXXXXX")"
  trap 'rm -f "${tmp}"' EXIT
  curl -fsSL --retry 3 -o "${tmp}" \
    "https://github.com/microsoft/onnxruntime/releases/download/v${version}/${asset}"
  got="$( (sha256sum "${tmp}" 2>/dev/null || shasum -a 256 "${tmp}") | cut -d' ' -f1)"
  if [ "${got}" != "${want}" ]; then
    echo "sha256 mismatch for ${asset}: got ${got}, want ${want}" >&2
    exit 1
  fi
  tar -xzf "${tmp}" -C "${cache}"
fi
echo "${dir}/lib/${lib}"
