//! Golden-output snapshot tests.
//!
//! Pins exact responses to catch changes in generation behavior.

use megahal::{GenerationLimit, MegaHal};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// A small, deterministic training corpus. Each sentence has at least 6 tokens
/// (the model order is 5) so that learning is not skipped per the spec.
const CORPUS: &[&str] = &[
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps in the sun while the fox runs",
    "dogs are loyal and friendly companions to humans",
    "foxes are clever and quick creatures of the forest",
    "humans love their dogs and cats and birds",
    "cats and dogs sometimes chase each other around the yard",
    "a quick brown fox is faster than a lazy dog",
    "the forest is full of foxes and deer and birds",
    "the brown fox and the lazy cat live in the forest together",
    "a quick dog and a slow fox both chase birds in the yard",
    "the sun shines brightly over the forest and the foxes hide in the trees",
    "loyal dogs sleep peacefully in the sun while their humans work nearby",
    "clever foxes and friendly dogs share the forest with deer and birds",
    "humans usually like their cats and dogs better than wild foxes",
    "the lazy cats sleep in the sun all day long every single day",
    "quick dogs chase lazy cats around the yard every morning at dawn",
    "dogs sleep in the sun and dream about chasing foxes through the forest",
    "birds and deer live in the forest near the brown fox and her cubs",
    "the fox jumps over the fence while the dogs bark loudly at her",
    "friendly humans feed the birds and the deer near the edge of the forest",
];

fn snapshot_hal(seed: u64) -> MegaHal<SmallRng> {
    let mut hal = MegaHal::new(5, SmallRng::seed_from_u64(seed));
    // Iteration cap, not timeout: keeps the test deterministic and fast.
    hal.set_limit(GenerationLimit::Iterations(50));
    for line in CORPUS {
        hal.learn(line);
    }
    hal
}

#[test]
fn snapshot_respond_dogs() {
    let mut hal = snapshot_hal(42);
    let reply = hal.respond("tell me about the dogs");
    // Updated: extract_keywords now returns keywords in input order (issue #4),
    // changing the seed distribution to match C's make_keywords/seed behavior.
    assert_eq!(
        reply,
        "Dogs are loyal and friendly dogs share the forest with deer and birds."
    );
}

#[test]
fn snapshot_respond_fox() {
    let mut hal = snapshot_hal(42);
    let reply = hal.respond("what is a fox");
    // Updated: see snapshot_respond_dogs.
    assert_eq!(
        reply,
        "A quick brown fox and the lazy dog sleeps in the sun while the fox runs."
    );
}

#[test]
fn snapshot_respond_forest() {
    let mut hal = snapshot_hal(42);
    let reply = hal.respond("describe the forest");
    // Updated: see snapshot_respond_dogs.
    assert_eq!(
        reply,
        "The forest is full of foxes and deer live in the sun while their humans work nearby."
    );
}

#[test]
fn snapshot_generate_does_not_learn() {
    let mut hal = snapshot_hal(42);
    let before_dict = hal.model().dictionary.len();
    let _ = hal.generate("tell me about the dogs");
    let after_dict = hal.model().dictionary.len();
    assert_eq!(
        before_dict, after_dict,
        "generate() must not mutate the dictionary"
    );
}

#[test]
fn snapshot_respond_unknown_input_falls_back() {
    // Input that's longer than `order` and contains words never seen in the
    // corpus. After learn-from-input the model has the input verbatim and
    // nothing else relevant; baseline equals input, fallback triggers.
    let mut hal = MegaHal::new(5, SmallRng::seed_from_u64(42));
    hal.set_limit(GenerationLimit::Iterations(20));
    let reply = hal.respond("xyzzy plugh frobozz are mystical incantations from days past");
    assert_eq!(reply, "I don't know enough to answer you yet!");
}

#[test]
fn snapshot_deterministic_across_runs() {
    // Sanity-check: two fresh instances with the same seed produce identical
    // output across all prompts above.
    let prompts = &[
        "tell me about the dogs",
        "what is a fox",
        "describe the forest",
    ];

    let mut a = snapshot_hal(42);
    let mut b = snapshot_hal(42);
    for p in prompts {
        assert_eq!(a.respond(p), b.respond(p));
    }
}
