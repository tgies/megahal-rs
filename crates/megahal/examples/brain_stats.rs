//! Print structural stats for a native (`MHALRUST`) brain file.
//!
//! Run with: `cargo run --example brain_stats -p megahal -- path/to/brain.brn`

use std::env;
use std::path::Path;
use std::process::ExitCode;

use megahal::MegaHal;
use rand::{SeedableRng, rngs::SmallRng};

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: brain_stats <path/to/brain.brn>");
        return ExitCode::from(2);
    };

    let mut hal = MegaHal::new(5, SmallRng::seed_from_u64(0));
    if let Err(e) = hal.load_brain(Path::new(&path)) {
        eprintln!("failed to load {path}: {e}");
        return ExitCode::FAILURE;
    }

    let model = hal.model();
    println!(
        "order={}, dictionary={}, forward_nodes={}, backward_nodes={}",
        model.order,
        model.dictionary.len(),
        model.forward.len(),
        model.backward.len(),
    );
    ExitCode::SUCCESS
}
