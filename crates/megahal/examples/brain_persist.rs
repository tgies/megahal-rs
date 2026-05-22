//! Save and load a trained brain through in-memory buffers, with no
//! filesystem involvement. Useful for games shipping a pre-trained model as
//! an embedded asset, for storing brains in a database, or for shuttling
//! state over a network.
//!
//! Run with: `cargo run --example brain_persist -p megahal`

use std::io::Cursor;

use megahal::{GenerationLimit, MegaHal};
use rand::{SeedableRng, rngs::SmallRng};

fn main() {
    let mut hal = MegaHal::new(5, SmallRng::seed_from_u64(7));
    hal.set_limit(GenerationLimit::Iterations(20));
    hal.train_from_reader(Cursor::new(
        b"the quick brown fox jumps over the lazy dog\n\
          dogs are loyal and friendly to humans\n\
          foxes live in the forest and chase birds\n"
            .as_slice(),
    ))
    .unwrap();

    // Serialize to a Vec<u8>.
    let mut blob = Vec::new();
    hal.save_brain_to_writer(&mut blob).unwrap();
    println!("serialized brain: {} bytes", blob.len());

    // Load a fresh instance from the same bytes.
    let mut restored = MegaHal::new(5, SmallRng::seed_from_u64(7));
    restored.set_limit(GenerationLimit::Iterations(20));
    restored
        .load_brain_from_reader(&mut Cursor::new(&blob))
        .unwrap();

    println!(
        "restored reply: {}",
        restored.respond("tell me about the fox")
    );
}
