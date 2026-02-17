//! MegaHAL conversational engine — a bidirectional Markov chain chatbot.
//!
//! This is the facade crate that wires together all the lower-level components:
//! - [`symbol_core`]: Symbol trait and SymbolId
//! - [`ngram_trie`]: Arena-based frequency trie
//! - [`symbol_dict`]: Interning dictionary
//! - [`markov_chain`]: Bidirectional model and context window
//! - [`megahal_tokenizer`]: Text tokenization
//! - [`megahal_keywords`]: Keyword extraction
//! - [`megahal_gen`]: Reply generation and evaluation
//!
//! # Quick Start
//!
//! ```
//! use megahal::MegaHal;
//! use rand::rngs::SmallRng;
//! use rand::SeedableRng;
//!
//! let mut hal = MegaHal::new(5, SmallRng::seed_from_u64(42));
//! hal.learn("The cat sat on the mat.");
//! let reply = hal.respond("Tell me about the cat.");
//! println!("{reply}");
//! ```

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::Path;

use markov_chain::BidirectionalModel;
use megahal_gen::{capitalize, generate_reply};
use megahal_tokenizer::tokenize;
use rand::Rng;
use symbol_core::Symbol;

// Re-export types that consumers (like the CLI) need.
pub use megahal_gen::GenerationLimit;
pub use megahal_keywords::{KeywordConfig, SwapTable, extract_keywords};

/// The MegaHAL symbol type: a case-insensitive byte string.
///
/// All comparisons are case-insensitive (both sides uppercased before comparison).
/// Ordering is lexicographic after uppercasing, with shorter strings comparing as
/// less-than if they share a prefix. This matches the original MegaHAL behavior.
#[derive(Clone, Debug, Hash)]
pub struct MegaHalSymbol(Vec<u8>);

impl MegaHalSymbol {
    /// Create a new symbol from a string (stored as uppercase bytes).
    pub fn new(s: &str) -> Self {
        MegaHalSymbol(s.to_uppercase().into_bytes())
    }

    /// Get the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Convert to a String (for display/output).
    pub fn to_string_lossy(&self) -> String {
        String::from_utf8_lossy(&self.0).into_owned()
    }

    /// Internal: uppercased bytes for comparison.
    fn upper(&self) -> Vec<u8> {
        self.0.iter().map(|b| b.to_ascii_uppercase()).collect()
    }
}

impl AsRef<[u8]> for MegaHalSymbol {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl PartialEq for MegaHalSymbol {
    fn eq(&self, other: &Self) -> bool {
        self.upper() == other.upper()
    }
}

impl Eq for MegaHalSymbol {}

impl PartialOrd for MegaHalSymbol {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MegaHalSymbol {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.upper().cmp(&other.upper())
    }
}

impl Symbol for MegaHalSymbol {
    fn error() -> Self {
        MegaHalSymbol(b"<ERROR>".to_vec())
    }

    fn fin() -> Self {
        MegaHalSymbol(b"<FIN>".to_vec())
    }
}

/// The MegaHAL conversational engine.
///
/// Generic over the PRNG type `R` for testability. Defaults to `SmallRng`
/// for efficient, seedable random generation.
pub struct MegaHal<R: Rng> {
    /// The bidirectional Markov model.
    model: BidirectionalModel<MegaHalSymbol>,
    /// Keyword extraction configuration (banned, auxiliary, swap).
    keyword_config: KeywordConfig,
    /// Auxiliary keyword set as MegaHalSymbol for generation.
    aux_symbols: HashSet<MegaHalSymbol>,
    /// Greeting keywords.
    greetings: Vec<String>,
    /// Generation loop limit.
    limit: GenerationLimit,
    /// Random number generator.
    rng: R,
}

impl<R: Rng> MegaHal<R> {
    /// Create a new MegaHAL engine with the given model order and PRNG.
    pub fn new(order: u8, rng: R) -> Self {
        MegaHal {
            model: BidirectionalModel::new(order),
            keyword_config: KeywordConfig::default(),
            aux_symbols: HashSet::new(),
            greetings: Vec::new(),
            limit: GenerationLimit::default(),
            rng,
        }
    }

    /// Set the generation limit.
    pub fn set_limit(&mut self, limit: GenerationLimit) {
        self.limit = limit;
    }

    /// Set keyword configuration (banned words, auxiliary words, swap table).
    pub fn set_keyword_config(&mut self, config: KeywordConfig) {
        // Build aux_symbols set for generation.
        self.aux_symbols = config
            .auxiliary
            .iter()
            .map(|s| MegaHalSymbol::new(s))
            .collect();
        self.keyword_config = config;
    }

    /// Set greeting keywords.
    pub fn set_greetings(&mut self, greetings: Vec<String>) {
        self.greetings = greetings;
    }

    /// Learn from an input string without generating a reply.
    pub fn learn(&mut self, input: &str) {
        let token_strings = tokenize(input);
        let tokens: Vec<MegaHalSymbol> = token_strings
            .iter()
            .map(|s| MegaHalSymbol::new(s))
            .collect();
        self.model.learn(&tokens);
    }

    /// Learn from input and generate a reply.
    ///
    /// This follows the MegaHAL conversation flow: learn first, then generate.
    pub fn respond(&mut self, input: &str) -> String {
        // Step 1: Tokenize.
        let token_strings = tokenize(input);
        let tokens: Vec<MegaHalSymbol> = token_strings
            .iter()
            .map(|s| MegaHalSymbol::new(s))
            .collect();

        // Step 2: Learn (before generating — matches original behavior).
        self.model.learn(&tokens);

        // Step 3: Extract keywords.
        let keywords = extract_keywords(
            &tokens,
            &self.model.dictionary,
            &self.keyword_config,
            MegaHalSymbol::new,
        );
        let keyword_symbols: HashSet<MegaHalSymbol> =
            keywords.iter().map(|s| MegaHalSymbol::new(s)).collect();

        // Step 4: Generate reply.
        let reply_symbols = generate_reply(
            &self.model,
            &tokens,
            &keyword_symbols,
            &self.aux_symbols,
            &self.limit,
            &mut self.rng,
        );

        // Step 5: Format output.
        if reply_symbols.is_empty() {
            return "I don't know enough to answer you yet!".to_string();
        }

        let reply_strings: Vec<String> =
            reply_symbols.iter().map(|s| s.to_string_lossy()).collect();
        capitalize(&reply_strings)
    }

    /// Generate an initial greeting (before any user input).
    pub fn greet(&mut self) -> String {
        if self.greetings.is_empty() {
            return "Hello!".to_string();
        }

        // Pick a random greeting keyword.
        let idx = self.rng.random_range(0..self.greetings.len());
        let greeting = self.greetings[idx].clone();

        let mut keywords = HashSet::new();
        keywords.insert(MegaHalSymbol::new(&greeting));

        let reply_symbols = generate_reply(
            &self.model,
            &[],
            &keywords,
            &self.aux_symbols,
            &self.limit,
            &mut self.rng,
        );

        if reply_symbols.is_empty() {
            return "Hello!".to_string();
        }

        let reply_strings: Vec<String> =
            reply_symbols.iter().map(|s| s.to_string_lossy()).collect();
        capitalize(&reply_strings)
    }

    /// Train from a text file (one sentence per line).
    pub fn train_from_file(&mut self, path: &Path) -> io::Result<()> {
        let content = fs::read_to_string(path)?;
        for line in content.lines() {
            let trimmed = line.trim();
            // Skip comments and empty lines.
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            self.learn(trimmed);
        }
        Ok(())
    }

    /// Get a reference to the underlying model (for inspection/testing).
    pub fn model(&self) -> &BidirectionalModel<MegaHalSymbol> {
        &self.model
    }
}

/// Load a keyword list file (one word per line, comments with #).
pub fn load_word_list(path: &Path) -> io::Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    Ok(content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.to_uppercase())
        .collect())
}

/// Load a swap file (space/tab-separated pairs, one per line).
pub fn load_swap_file(path: &Path) -> io::Result<Vec<(String, String)>> {
    let content = fs::read_to_string(path)?;
    Ok(content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let from = parts.next()?.to_uppercase();
            let to = parts.next()?.to_uppercase();
            Some((from, to))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn test_hal() -> MegaHal<SmallRng> {
        MegaHal::new(5, SmallRng::seed_from_u64(42))
    }

    fn trained_hal() -> MegaHal<SmallRng> {
        let mut hal = test_hal();
        for _ in 0..5 {
            hal.learn("The quick brown fox jumps over the lazy dog.");
            hal.learn("Dogs are wonderful animals that bring joy to people.");
            hal.learn("Cats and dogs are popular pets around the world.");
        }
        hal.set_limit(GenerationLimit::Iterations(20));
        hal
    }

    // --- MegaHalSymbol tests ---

    #[test]
    fn megahal_symbol_case_insensitive() {
        let a = MegaHalSymbol::new("Hello");
        let b = MegaHalSymbol::new("HELLO");
        let c = MegaHalSymbol::new("hello");
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn megahal_symbol_sentinels() {
        let error = MegaHalSymbol::error();
        let fin = MegaHalSymbol::fin();
        assert_ne!(error, fin);
    }

    #[test]
    fn megahal_symbol_as_ref() {
        let sym = MegaHalSymbol::new("TEST");
        let bytes: &[u8] = sym.as_ref();
        assert_eq!(bytes, b"TEST");
    }

    #[test]
    fn megahal_symbol_ordering() {
        let apple = MegaHalSymbol::new("apple");
        let banana = MegaHalSymbol::new("BANANA");
        assert!(apple < banana);
    }

    #[test]
    fn megahal_symbol_hash_case_insensitive() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MegaHalSymbol::new("Hello"));
        assert!(set.contains(&MegaHalSymbol::new("HELLO")));
        assert!(set.contains(&MegaHalSymbol::new("hello")));
    }

    #[test]
    fn megahal_symbol_to_string_lossy() {
        let sym = MegaHalSymbol::new("Hello");
        assert_eq!(sym.to_string_lossy(), "HELLO");
    }

    // --- Engine lifecycle tests ---

    #[test]
    fn new_engine_creates_empty_model() {
        let hal = test_hal();
        assert_eq!(hal.model().dictionary.len(), 2); // just sentinels
    }

    #[test]
    fn learn_adds_to_dictionary() {
        let mut hal = test_hal();
        hal.learn("The cat sat on the mat.");
        assert!(hal.model().dictionary.len() > 2);
    }

    #[test]
    fn learn_multiple_sentences_grows_dict() {
        let mut hal = test_hal();
        hal.learn("The cat sat.");
        let after_first = hal.model().dictionary.len();
        hal.learn("A new dog ran.");
        let after_second = hal.model().dictionary.len();
        assert!(after_second > after_first);
    }

    // --- respond tests ---

    #[test]
    fn respond_returns_non_empty() {
        let mut hal = trained_hal();
        let reply = hal.respond("Tell me about dogs.");
        assert!(!reply.is_empty());
    }

    #[test]
    fn respond_learns_before_generating() {
        let mut hal = test_hal();
        // No training data at all. First respond call should learn the input.
        hal.set_limit(GenerationLimit::Iterations(5));
        let _ = hal.respond("The cat sat on the mat and looked at the world.");
        // After responding, the model should have learned the input tokens.
        assert!(hal.model().dictionary.len() > 2);
    }

    #[test]
    fn respond_deterministic_with_same_seed() {
        let build = || {
            let mut hal = trained_hal();
            hal.respond("Tell me about cats.")
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn respond_returns_canned_when_empty() {
        let mut hal = test_hal();
        // Very short input with no training → model can't generate.
        hal.set_limit(GenerationLimit::Iterations(5));
        let reply = hal.respond("Hi.");
        // Should return the canned fallback message.
        assert_eq!(reply, "I don't know enough to answer you yet!");
    }

    // --- Keyword config tests ---

    #[test]
    fn set_keyword_config_builds_aux_symbols() {
        let mut hal = test_hal();
        let mut config = KeywordConfig::default();
        config.auxiliary.insert("MY".into());
        config.auxiliary.insert("YOUR".into());
        hal.set_keyword_config(config);
        assert_eq!(hal.aux_symbols.len(), 2);
    }

    #[test]
    fn respond_with_banned_words() {
        let mut hal = trained_hal();
        let mut config = KeywordConfig::default();
        config.banned.insert("THE".into());
        config.banned.insert("ON".into());
        hal.set_keyword_config(config);
        // Should still generate a reply even with banned words.
        let reply = hal.respond("The cat.");
        assert!(!reply.is_empty());
    }

    // --- Greeting tests ---

    #[test]
    fn greet_without_training_returns_hello() {
        let mut hal = test_hal();
        assert_eq!(hal.greet(), "Hello!");
    }

    #[test]
    fn greet_with_empty_greetings_returns_hello() {
        let mut hal = test_hal();
        hal.set_greetings(vec![]);
        assert_eq!(hal.greet(), "Hello!");
    }

    #[test]
    fn greet_with_greetings_and_training() {
        let mut hal = trained_hal();
        hal.set_greetings(vec!["DOGS".into()]);
        let greeting = hal.greet();
        // With training data about dogs and "DOGS" as greeting keyword,
        // should produce something (may fallback to "Hello!" if generation fails).
        assert!(!greeting.is_empty());
    }

    // --- Generation limit tests ---

    #[test]
    fn set_limit_iterations() {
        let mut hal = trained_hal();
        hal.set_limit(GenerationLimit::Iterations(1));
        let reply = hal.respond("Tell me about foxes.");
        assert!(!reply.is_empty());
    }

    #[test]
    fn set_limit_both() {
        let mut hal = trained_hal();
        hal.set_limit(GenerationLimit::Both {
            timeout: std::time::Duration::from_millis(100),
            max_iterations: 5,
        });
        let reply = hal.respond("Tell me about foxes.");
        assert!(!reply.is_empty());
    }

    // --- File loading tests ---

    #[test]
    fn load_word_list_parses_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("megahal_test_load_word_list.txt");
        fs::write(&path, "# comment\nHELLO\nworld\n\n# another\nFOO\n").unwrap();
        let words = load_word_list(&path).unwrap();
        assert_eq!(words, vec!["HELLO", "WORLD", "FOO"]);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn load_swap_file_parses_pairs() {
        let dir = std::env::temp_dir();
        let path = dir.join("megahal_test_load_swap_file.txt");
        fs::write(&path, "# comment\nI\tYOU\nMY YOUR\n").unwrap();
        let pairs = load_swap_file(&path).unwrap();
        assert_eq!(
            pairs,
            vec![
                ("I".to_string(), "YOU".to_string()),
                ("MY".to_string(), "YOUR".to_string()),
            ]
        );
        fs::remove_file(&path).ok();
    }

    #[test]
    fn train_from_file_learns() {
        let dir = std::env::temp_dir();
        let path = dir.join("megahal_test_train_from_file.txt");
        fs::write(
            &path,
            "# comment\nThe cat sat on the mat.\nDogs are nice animals that play.\n",
        )
        .unwrap();
        let mut hal = test_hal();
        hal.train_from_file(&path).unwrap();
        assert!(hal.model().dictionary.len() > 2);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn train_from_file_skips_comments_and_blanks() {
        let dir = std::env::temp_dir();
        let path = dir.join("megahal_test_train_comments.txt");
        fs::write(&path, "# this is a comment\n\n# another comment\n").unwrap();
        let mut hal = test_hal();
        hal.train_from_file(&path).unwrap();
        // Only comments and blanks → nothing learned.
        assert_eq!(hal.model().dictionary.len(), 2);
        fs::remove_file(&path).ok();
    }
}
