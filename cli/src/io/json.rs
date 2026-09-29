use std::fs;
use std::path::Path;

use anyhow::Result;
use serde::Serialize;

pub fn print_or_write<T: Serialize>(value: &T, output: Option<&Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    match output {
        Some(path) => {
            fs::write(path, &json)?;
            println!("wrote to {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}
