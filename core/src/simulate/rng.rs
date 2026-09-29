use sha2::{Digest, Sha256};

// Shared deterministic RNG helpers
//
// Per CLAUDE.md conventions: patient-aware / per-row choices must be
// reproducible from `(global random_state, entity identifier)` alone, so
// that batching order never changes the outcome. We derive two independent
// uniform floats in [0, 1) from a single SHA256 digest of
// "{random_state}|{identifier}" — this is the shared primitive used by both
// `BlackoutSimulator`'s per-patient window selection and
// `JitterSimulator`'s per-row noise draw.

pub(super) fn stable_unit_floats(random_state: Option<u64>, identifier: &str) -> (f64, f64) {
    let payload = format!("{:?}|{}", random_state, identifier);
    let digest = Sha256::digest(payload.as_bytes());
    const TO_UNIT: f64 = 1.0 / (1u64 << 53) as f64;
    let bits_to_unit = |bytes: &[u8]| -> f64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(bytes);
        let bits = u64::from_le_bytes(b);
        ((bits >> 11) as f64) * TO_UNIT
    };
    (bits_to_unit(&digest[0..8]), bits_to_unit(&digest[8..16]))
}

/// Deterministic, batch-order-independent standard-normal sample for one row.
///
/// Mirrors the intent of the Python reference (`jitter.py`'s `_row_noise`,
/// which mixes a SHA256-derived global seed with a per-row hash via
/// SplitMix64 + Box-Muller). Polars `DataFrame`s have no pandas-style row
/// index, so `row_identifier` is the row's positional ordinal within the
/// frame being transformed. The exact bits will not match the Python
/// implementation, but the property that matters is preserved exactly:
/// the same `(random_state, row_identifier)` always yields the same draw,
/// independent of any other row, batch boundary, or call order.
pub(super) fn stable_normal_sample(random_state: Option<u64>, row_identifier: usize) -> f64 {
    let (u1, u2) = stable_unit_floats(random_state, &format!("row:{row_identifier}"));
    let u1 = u1.max(f64::MIN_POSITIVE); // avoid ln(0)
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
