//! Load a C MegaHAL `MegaHALv8` brain file and generate a reply from it.
//!
//! Pass the path to a `.brn` file as the first argument. Useful for
//! validating an importer against brains trained by the original C MegaHAL.
//!
//! Run with: `cargo run --example load_v8_brain -p megahal -- path/to/megahal.brn`

use std::env;
use std::path::Path;
use std::process::ExitCode;

use megahal::{GenerationLimit, MegaHal};
use rand::{SeedableRng, rngs::SmallRng};

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: load_v8_brain <path/to/megahal.brn>");
        return ExitCode::from(2);
    };

    let mut hal = MegaHal::new(5, SmallRng::seed_from_u64(0));
    hal.set_limit(GenerationLimit::Iterations(20));

    if let Err(e) = hal.load_v8_brain(Path::new(&path)) {
        eprintln!("failed to load {path}: {e}");
        return ExitCode::FAILURE;
    }

    let model = hal.model();
    println!(
        "loaded brain: order={}, dict_size={}, forward_nodes={}, backward_nodes={}",
        model.order,
        model.dictionary.len(),
        model.forward.len(),
        model.backward.len(),
    );

    println!("reply: {}", hal.respond("hello there"));
    ExitCode::SUCCESS
}
