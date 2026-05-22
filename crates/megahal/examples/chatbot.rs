//! A minimal interactive chatbot. Trains on a tiny hardcoded corpus, then
//! reads lines from stdin and replies to each one until EOF.
//!
//! Run with: `cargo run --example chatbot -p megahal`

use std::io::{self, BufRead, Write};

use megahal::{GenerationLimit, MegaHal};
use rand::{SeedableRng, rngs::SmallRng};

const CORPUS: &[&str] = &[
    "the quick brown fox jumps over the lazy dog",
    "dogs are loyal and friendly companions to humans",
    "foxes are clever and quick creatures of the forest",
    "humans love their dogs and cats and birds",
    "the forest is full of foxes and deer and birds",
    "cats and dogs sometimes chase each other around the yard",
];

fn main() -> io::Result<()> {
    let mut hal = MegaHal::new(5, SmallRng::seed_from_u64(0xC0FFEE));
    hal.set_limit(GenerationLimit::Iterations(50));
    for line in CORPUS {
        hal.learn(line);
    }

    println!("{}", hal.greet());
    println!("(Ctrl-D to exit)");

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        writeln!(stdout, "{}", hal.respond(input))?;
        stdout.flush()?;
    }
    Ok(())
}
