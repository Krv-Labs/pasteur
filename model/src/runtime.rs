//! Finding ONNX Runtime.
//!
//! With the default `load-dynamic` feature, ONNX Runtime is a shared library
//! opened at run time rather than a static archive linked at build time. The
//! archive `ort` downloads targets glibc 2.38 and GCC 13's libstdc++, so it
//! does not link on Ubuntu 22.04, Debian 12, or RHEL 8/9; Microsoft's own
//! release builds need only glibc 2.27.
//!
//! Search order, first hit wins:
//!
//! 1. `ORT_DYLIB_PATH`, if set. Explicit beats discovered: when it is set and
//!    does not load, that is the error, with no fallback.
//! 2. `lib/` beside the executable's directory, the release tarball layout
//!    (`bin/pasteur-cli`, `lib/libonnxruntime.so*`).
//! 3. The bare library name, which `ort` resolves next to the executable and
//!    then through the system loader (`LD_LIBRARY_PATH`, `ldconfig`).

use pasteur_core::CoreError;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Environment variable naming the ONNX Runtime shared library to load.
pub const ORT_DYLIB_ENV: &str = "ORT_DYLIB_PATH";

/// Oldest ONNX Runtime whose C API this build calls (`ort::MINOR_VERSION`).
pub const MIN_ORT_VERSION: &str = "1.17";

#[cfg(target_os = "macos")]
const LIB_NAMES: &[&str] = &["libonnxruntime.dylib", "libonnxruntime.1.dylib"];
#[cfg(target_os = "windows")]
const LIB_NAMES: &[&str] = &["onnxruntime.dll"];
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const LIB_NAMES: &[&str] = &["libonnxruntime.so", "libonnxruntime.so.1"];

/// Which library was loaded, for `pasteur-cli doctor` and error reports.
#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    pub path: PathBuf,
    pub build: String,
}

static RUNTIME: OnceLock<Result<RuntimeInfo, String>> = OnceLock::new();

/// Load ONNX Runtime once per process. Every model load calls this first, so
/// a missing library is an error naming the fixes, not a panic deep in `ort`.
pub fn ensure() -> Result<RuntimeInfo, CoreError> {
    RUNTIME
        .get_or_init(load)
        .clone()
        .map_err(CoreError::InvalidConfig)
}

/// The places [`ensure`] tries, in order, for the given environment.
pub fn candidates(env_path: Option<&str>, exe: Option<&Path>) -> Vec<PathBuf> {
    if let Some(path) = env_path.filter(|p| !p.is_empty()) {
        return vec![PathBuf::from(path)];
    }
    let mut out = Vec::new();
    if let Some(lib_dir) = exe.and_then(Path::parent).and_then(Path::parent) {
        let lib_dir = lib_dir.join("lib");
        out.extend(LIB_NAMES.iter().map(|name| lib_dir.join(name)));
    }
    out.extend(LIB_NAMES.iter().map(PathBuf::from));
    out
}

#[cfg(feature = "load-dynamic")]
fn load() -> Result<RuntimeInfo, String> {
    let env_path = std::env::var(ORT_DYLIB_ENV).ok();
    let exe = std::env::current_exe().ok();
    let mut tried = Vec::new();
    for path in candidates(env_path.as_deref(), exe.as_deref()) {
        // `ort::init_from` may run only once per process: in ort 2.0.0-rc.13 a
        // failed load still marks its library slot initialized, and the next
        // call reads that slot uninitialized. So each candidate is opened and
        // checked here first, and ort is handed the one that works.
        let path = beside_exe(&path, exe.as_deref());
        match probe(&path) {
            Ok(version) => {
                ort::init_from(&path)
                    .map_err(|e| format!("{}: {e}", path.display()))?
                    .commit();
                return Ok(RuntimeInfo {
                    path,
                    build: format!("ONNX Runtime {version} ({})", ort::info()),
                });
            }
            Err(err) => tried.push(format!("{}: {err}", path.display())),
        }
    }
    Err(not_found(env_path.as_deref(), &tried))
}

/// `ort` resolves a bare library name next to the executable before asking
/// the system loader; resolve it the same way so the probe opens what ort will.
#[cfg(feature = "load-dynamic")]
fn beside_exe(path: &Path, exe: Option<&Path>) -> PathBuf {
    if path.is_relative() {
        if let Some(local) = exe.and_then(Path::parent).map(|d| d.join(path)) {
            if local.exists() {
                return local;
            }
        }
    }
    path.to_path_buf()
}

/// Open the library, confirm it is ONNX Runtime, and return its version if it
/// is new enough for this build's C API.
#[cfg(feature = "load-dynamic")]
fn probe(path: &Path) -> Result<String, String> {
    if path.is_absolute() && !path.exists() {
        return Err("not found".into());
    }
    // SAFETY: loading a shared library runs its initializers; the candidates
    // are ONNX Runtime install locations the user configured or ships.
    let lib = unsafe { libloading::Library::new(path) }.map_err(|e| {
        // libloading's Display is just "dlopen failed"; the reason is the source.
        match std::error::Error::source(&e) {
            Some(why) => why.to_string(),
            None => e.to_string(),
        }
    })?;
    let get_base: libloading::Symbol<unsafe extern "C" fn() -> *const ort::sys::OrtApiBase> =
        unsafe { lib.get(b"OrtGetApiBase") }
            .map_err(|_| "not ONNX Runtime (no OrtGetApiBase)".to_string())?;
    // SAFETY: OrtGetApiBase is ONNX Runtime's stable C entry point; it returns
    // a static struct whose GetVersionString returns a static C string.
    let version = unsafe {
        let base = get_base();
        if base.is_null() {
            return Err("OrtGetApiBase returned null".into());
        }
        std::ffi::CStr::from_ptr(((*base).GetVersionString)())
            .to_string_lossy()
            .into_owned()
    };
    if new_enough(&version, ort::MINOR_VERSION) {
        Ok(version)
    } else {
        Err(format!("version {version} is older than {MIN_ORT_VERSION}"))
    }
}

/// ONNX Runtime is 1.x; this build needs at least `1.{min_minor}`.
#[cfg(feature = "load-dynamic")]
fn new_enough(version: &str, min_minor: u32) -> bool {
    let mut parts = version.split('.').map(|p| p.parse::<u32>().ok());
    match (parts.next().flatten(), parts.next().flatten()) {
        (Some(1), Some(minor)) => minor >= min_minor,
        (Some(major), Some(_)) => major > 1,
        _ => false,
    }
}

#[cfg(not(feature = "load-dynamic"))]
fn load() -> Result<RuntimeInfo, String> {
    Ok(RuntimeInfo {
        path: PathBuf::from("(statically linked)"),
        build: ort::info().to_string(),
    })
}

#[cfg(feature = "load-dynamic")]
fn not_found(env_path: Option<&str>, tried: &[String]) -> String {
    let bullets = |items: &mut dyn Iterator<Item = &str>| -> String {
        items.map(|t| format!("\n  - {t}")).collect()
    };
    let lead = match env_path {
        Some(p) if !p.is_empty() => format!("{ORT_DYLIB_ENV}={p} did not load:"),
        _ => format!(
            "ONNX Runtime >= {MIN_ORT_VERSION} was not found. pasteur-cli loads it at run time; tried:"
        ),
    };
    format!(
        "{lead}{}\nFix one of:{}",
        bullets(&mut tried.iter().map(String::as_str)),
        bullets(&mut FIXES.iter().copied()),
    )
}

#[cfg(feature = "load-dynamic")]
/// How to get a library onto the search path, most convenient first.
const FIXES: &[&str] = &["download onnxruntime-<os>-<arch>-<version>.tgz from \
     https://github.com/microsoft/onnxruntime/releases and set ORT_DYLIB_PATH to \
     the shared library in its lib/ (on Linux, lib/libonnxruntime.so)"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_path_is_the_only_candidate() {
        let got = candidates(
            Some("/opt/ort/libx.so"),
            Some(Path::new("/x/bin/pasteur-cli")),
        );
        assert_eq!(got, vec![PathBuf::from("/opt/ort/libx.so")]);
    }

    #[test]
    fn empty_env_falls_through_to_discovery() {
        let got = candidates(Some(""), None);
        assert_eq!(got.len(), LIB_NAMES.len());
        assert!(got.iter().all(|p| p.is_relative()));
    }

    #[test]
    fn release_layout_comes_before_the_system_loader() {
        let got = candidates(None, Some(Path::new("/opt/pasteur/bin/pasteur-cli")));
        assert_eq!(got[0], Path::new("/opt/pasteur/lib").join(LIB_NAMES[0]));
        assert_eq!(
            got.last().unwrap(),
            &PathBuf::from(LIB_NAMES[LIB_NAMES.len() - 1])
        );
    }

    #[cfg(feature = "load-dynamic")]
    #[test]
    fn version_floor_is_the_c_api_minor() {
        assert!(new_enough("1.17.0", 17));
        assert!(new_enough("1.28.0", 17));
        assert!(!new_enough("1.16.3", 17));
        assert!(new_enough("2.0.0", 17));
        assert!(!new_enough("garbage", 17));
        assert_eq!(MIN_ORT_VERSION, format!("1.{}", ort::MINOR_VERSION));
    }

    #[cfg(feature = "load-dynamic")]
    #[test]
    fn not_found_names_every_fix() {
        let msg = not_found(None, &["a: not found".into()]);
        assert!(msg.contains("a: not found"));
        assert!(msg.contains(ORT_DYLIB_ENV));
        assert!(msg.contains("github.com/microsoft/onnxruntime/releases"));
        let msg = not_found(Some("/bad.so"), &[]);
        assert!(msg.starts_with("ORT_DYLIB_PATH=/bad.so did not load"));
    }
}
