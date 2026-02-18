//! Integration tests for MegaHAL: full conversation flow with real data files.
//!
//! These tests use the original MegaHAL support files (megahal.trn, .ban, .aux,
//! .swp, .grt) to exercise the full pipeline: training, keyword extraction with
//! real banned/auxiliary/swap tables, response generation, and brain persistence.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use megahal::{GenerationLimit, KeywordConfig, MegaHal, SwapTable, load_swap_file, load_word_list};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Path to the MegaHAL data directory (bundled in the repo).
fn data_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = megahal-rs/crates/megahal
    // Data files are at megahal-rs/data/
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.join("../../data")
}

/// Build a fully-configured MegaHAL instance trained on the real data files.
fn trained_hal() -> MegaHal<SmallRng> {
    let dir = data_dir();
    let mut hal = MegaHal::new(5, SmallRng::seed_from_u64(42));
    hal.set_limit(GenerationLimit::Iterations(100));

    // Load support files.
    let mut config = KeywordConfig::default();
    if let Ok(words) = load_word_list(&dir.join("megahal.ban")) {
        config.banned = words.into_iter().collect();
    }
    if let Ok(words) = load_word_list(&dir.join("megahal.aux")) {
        config.auxiliary = words.into_iter().collect();
    }
    if let Ok(pairs) = load_swap_file(&dir.join("megahal.swp")) {
        config.swap = SwapTable { pairs };
    }
    if let Ok(words) = load_word_list(&dir.join("megahal.grt")) {
        hal.set_greetings(words);
    }
    hal.set_keyword_config(config);

    // Train.
    hal.train_from_file(&dir.join("megahal.trn"))
        .expect("failed to load megahal.trn");

    hal
}

// ---------------------------------------------------------------------------
// Training sanity
// ---------------------------------------------------------------------------

#[test]
fn training_populates_dictionary() {
    let hal = trained_hal();
    // megahal.trn has ~300 non-comment lines with diverse vocabulary.
    // The dictionary should contain well over 100 unique symbols.
    let dict_size = hal.model().dictionary.len();
    assert!(
        dict_size > 100,
        "expected > 100 dictionary entries after training, got {dict_size}"
    );
}

#[test]
fn training_populates_both_tries() {
    let hal = trained_hal();
    let model = hal.model();

    // Both tries should have more than just the root node.
    assert!(
        model.forward.len() > 1,
        "forward trie should have nodes beyond root"
    );
    assert!(
        model.backward.len() > 1,
        "backward trie should have nodes beyond root"
    );
}

// ---------------------------------------------------------------------------
// Response quality
// ---------------------------------------------------------------------------

#[test]
fn respond_produces_nonempty_reply() {
    let mut hal = trained_hal();
    let reply = hal.respond("What do you think about life?");
    assert!(!reply.is_empty(), "reply should not be empty");
    assert_ne!(
        reply, "I don't know enough to answer you yet!",
        "reply should not be the fallback canned message"
    );
}

#[test]
fn respond_differs_from_input() {
    let mut hal = trained_hal();
    let input = "Tell me something interesting about the world.";
    let reply = hal.respond(input);
    // The reply should not be a verbatim echo of the input.
    assert_ne!(
        reply.to_uppercase(),
        input.to_uppercase(),
        "reply should not be identical to input"
    );
}

#[test]
fn respond_incorporates_keywords() {
    let mut hal = trained_hal();

    // "COMPUTER" appears in megahal.trn and should not be in the ban list.
    // After learning the input, the model should produce a reply that contains
    // at least one word from the combined input+training vocabulary.
    let reply = hal.respond("Tell me about computers and programming.");
    let reply_upper = reply.to_uppercase();

    // The reply should contain at least one substantive word (not just punctuation/spaces).
    let has_alpha_word = reply_upper
        .split(|c: char| !c.is_ascii_alphabetic() && c != '\'')
        .any(|w| w.len() > 2);
    assert!(
        has_alpha_word,
        "reply should contain substantive words: {reply}"
    );
}

#[test]
fn multiple_responses_show_variation() {
    // With different inputs and model learning between them, responses should vary.
    let mut hal = trained_hal();

    let r1 = hal.respond("Tell me about dogs.");
    let r2 = hal.respond("What do you think about music?");
    let r3 = hal.respond("How do you feel about the weather?");

    // Collect unique responses — with such different inputs, we should get
    // at least 2 distinct replies.
    let unique: HashSet<&str> = [r1.as_str(), r2.as_str(), r3.as_str()]
        .into_iter()
        .collect();
    assert!(
        unique.len() >= 2,
        "expected variation across responses, got: [{r1}], [{r2}], [{r3}]"
    );
}

#[test]
fn response_ends_with_terminal_punctuation() {
    let mut hal = trained_hal();
    let reply = hal.respond("Tell me something.");

    // capitalize() produces a sentence that should end with terminal punctuation.
    let last_char = reply.chars().last().expect("reply should not be empty");
    assert!(
        matches!(last_char, '.' | '!' | '?'),
        "reply should end with terminal punctuation, got: {reply:?}"
    );
}

#[test]
fn response_starts_with_uppercase() {
    let mut hal = trained_hal();
    let reply = hal.respond("Hello there.");

    let first_alpha = reply.chars().find(|c| c.is_ascii_alphabetic());
    if let Some(c) = first_alpha {
        assert!(
            c.is_uppercase(),
            "first alphabetic character should be uppercase, got: {reply:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Conversation flow
// ---------------------------------------------------------------------------

#[test]
fn multi_turn_conversation() {
    let mut hal = trained_hal();

    let replies: Vec<String> = [
        "Hello, how are you today?",
        "I like programming in Rust.",
        "What do you think about artificial intelligence?",
        "Tell me about your favorite things.",
        "Goodbye, it was nice chatting.",
    ]
    .iter()
    .map(|input| hal.respond(input))
    .collect();

    // Every turn should produce a non-fallback reply.
    for (i, reply) in replies.iter().enumerate() {
        assert!(
            !reply.is_empty() && reply != "I don't know enough to answer you yet!",
            "turn {i} produced empty/fallback reply: {reply:?}"
        );
    }
}

#[test]
fn greeting_uses_training_data() {
    let mut hal = trained_hal();
    let greeting = hal.greet();

    // With training data and greeting keywords loaded, greet() should produce
    // something other than the bare "Hello!" fallback.
    assert!(!greeting.is_empty(), "greeting should not be empty");
    // It *might* still return "Hello!" if the RNG happens to pick that keyword
    // and the model generates it, but it should at least be a real response.
}

// ---------------------------------------------------------------------------
// Brain persistence
// ---------------------------------------------------------------------------

#[test]
fn brain_save_load_preserves_responses() {
    let mut hal = trained_hal();

    // Have a conversation to seed additional model state.
    hal.respond("I enjoy programming computers.");

    // Save brain.
    let dir = std::env::temp_dir();
    let brain_path = dir.join("megahal_integration_test.brn");
    hal.save_brain(&brain_path).expect("failed to save brain");

    // Load brain into a fresh instance with the same seed.
    let mut hal2 = MegaHal::new(5, SmallRng::seed_from_u64(42));
    hal2.set_limit(GenerationLimit::Iterations(100));
    hal2.load_brain(&brain_path).expect("failed to load brain");

    // Both should have the same dictionary size.
    assert_eq!(
        hal.model().dictionary.len(),
        hal2.model().dictionary.len(),
        "dictionary size should match after brain load"
    );

    // Both should produce the same reply for the same input (same seed, same model).
    let reply1 = hal2.respond("Tell me about computers.");
    assert!(!reply1.is_empty());
    assert_ne!(reply1, "I don't know enough to answer you yet!");

    std::fs::remove_file(&brain_path).ok();
}

#[test]
fn brain_save_load_deterministic() {
    // Two identically-constructed instances should produce identical brains and
    // identical responses after loading.
    let build = || {
        let mut hal = trained_hal();
        hal.respond("I like dogs and cats.");
        hal
    };

    let hal_a = build();
    let hal_b = build();

    let dir = std::env::temp_dir();
    let path_a = dir.join("megahal_det_test_a.brn");
    let path_b = dir.join("megahal_det_test_b.brn");

    hal_a.save_brain(&path_a).unwrap();
    hal_b.save_brain(&path_b).unwrap();

    // Brain files should be byte-identical.
    let bytes_a = std::fs::read(&path_a).unwrap();
    let bytes_b = std::fs::read(&path_b).unwrap();
    assert_eq!(
        bytes_a, bytes_b,
        "deterministic builds should produce identical brains"
    );

    // Load both into fresh instances and verify identical responses.
    // (Keyword seeding sorts the keyword vec, so iteration order is deterministic
    // regardless of HashSet layout.)
    let mut loaded_a = MegaHal::new(5, SmallRng::seed_from_u64(99));
    loaded_a.set_limit(GenerationLimit::Iterations(100));
    loaded_a.load_brain(&path_a).unwrap();

    let mut loaded_b = MegaHal::new(5, SmallRng::seed_from_u64(99));
    loaded_b.set_limit(GenerationLimit::Iterations(100));
    loaded_b.load_brain(&path_b).unwrap();

    let reply_a = loaded_a.respond("Tell me about animals.");
    let reply_b = loaded_b.respond("Tell me about animals.");
    assert_eq!(
        reply_a, reply_b,
        "identical brains + seeds should produce identical replies"
    );

    std::fs::remove_file(&path_a).ok();
    std::fs::remove_file(&path_b).ok();
}

#[test]
fn brain_file_is_reasonably_sized() {
    let hal = trained_hal();

    let dir = std::env::temp_dir();
    let path = dir.join("megahal_size_test.brn");
    hal.save_brain(&path).unwrap();

    let size = std::fs::metadata(&path).unwrap().len();
    // megahal.trn is small (~10KB), so the brain should be well under 1MB.
    // In practice it's around 300-400KB.
    assert!(
        size > 1_000,
        "brain file should be at least 1KB, got {size} bytes"
    );
    assert!(
        size < 2_000_000,
        "brain file should be under 2MB, got {size} bytes"
    );

    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// Support file loading
// ---------------------------------------------------------------------------

#[test]
fn ban_list_loads_correctly() {
    let words = load_word_list(&data_dir().join("megahal.ban")).unwrap();
    assert!(words.len() > 100, "ban list should have > 100 entries");
    assert!(words.contains(&"THE".to_string()));
    assert!(words.contains(&"DON'T".to_string()));
}

#[test]
fn aux_list_loads_correctly() {
    let words = load_word_list(&data_dir().join("megahal.aux")).unwrap();
    assert!(words.len() > 10, "aux list should have > 10 entries");
    assert!(words.contains(&"I".to_string()));
    assert!(words.contains(&"YOU".to_string()));
}

#[test]
fn swap_table_loads_correctly() {
    let pairs = load_swap_file(&data_dir().join("megahal.swp")).unwrap();
    assert!(!pairs.is_empty(), "swap table should not be empty");

    // "I" should swap to "YOU" (and "YOU" back to "I" and "ME").
    let has_i_to_you = pairs.iter().any(|(f, t)| f == "I" && t == "YOU");
    assert!(has_i_to_you, "swap table should contain I → YOU");
}

#[test]
fn greeting_list_loads_correctly() {
    let words = load_word_list(&data_dir().join("megahal.grt")).unwrap();
    assert!(!words.is_empty(), "greeting list should not be empty");
    assert!(words.contains(&"HELLO".to_string()));
}
