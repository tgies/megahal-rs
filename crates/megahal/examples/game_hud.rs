//! Embedding MegaHAL in a game HUD: simulate a 60 fps loop where each frame
//! has a few milliseconds to produce a line of "robot chatter" for an NPC.
//!
//! Run with: `cargo run --example game_hud -p megahal`

use std::time::{Duration, Instant};

use megahal::{GenerationLimit, MegaHal};
use rand::{SeedableRng, rngs::SmallRng};

const LORE: &[&str] = &[
    "the reactor core is stable at nominal output levels",
    "warning unauthorized presence detected in sector seven",
    "all systems are functioning within expected parameters",
    "the perimeter scanner shows movement near the eastern gate",
    "communications relay is online and broadcasting on all frequencies",
    "diagnostics complete no anomalies were detected in this cycle",
];

fn main() {
    let mut bot = MegaHal::new(5, SmallRng::seed_from_u64(0xDEADBEEF));

    // Tight per-frame budget. Iterations cap is what matters for `wasm32`
    // and other targets where `Instant::now()` may not be available;
    // including a timeout too gives a wall-clock safety net.
    bot.set_limit(GenerationLimit::Both {
        timeout: Duration::from_millis(4),
        max_iterations: 16,
    });

    for line in LORE {
        bot.learn(line);
    }

    // Simulate 5 frames of HUD updates.
    for frame in 0..5 {
        let started = Instant::now();
        // `generate` rather than `respond` so player-supplied input doesn't
        // train the model.
        let chatter = bot
            .generate("system status")
            .unwrap_or_else(|| "[static]".into());
        let elapsed = started.elapsed();
        println!("frame {frame:>2} ({elapsed:>5.1?}): {chatter}");
    }
}
