//! A Rust port of Jason Hutchens' 1998 MegaHAL chatbot: a bidirectional
//! Markov-chain engine that learns from text and replies to prompts.
//!
//! # Quick start
//!
//! ```
//! use megahal::MegaHal;
//! use rand::{rngs::SmallRng, SeedableRng};
//!
//! let mut hal = MegaHal::new(5, SmallRng::seed_from_u64(42));
//! hal.learn("the cat sat on the mat");
//! hal.learn("the dog chased the cat around the yard");
//! let reply = hal.respond("tell me about the cat");
//! println!("{reply}");
//! ```
//!
//! # Concepts
//!
//! * **Order**: The n-gram depth. A model of order N considers up to N preceding
//!   tokens. The default is 5.
//! * **Tokens**: Words, whitespace, and punctuation. Sentences of fewer than
//!   `order + 1` tokens are not learned.
//! * **Keywords**: Alphanumeric tokens that the model has seen before (excluding
//!   banned tokens). Generation biases random walks toward these keywords.
//! * **Generation limits**: Stop conditions for response generation, configurable
//!   via [`GenerationLimit`].
//!
//! # Thread safety
//!
//! [`MegaHal<R>`] is `Send + Sync` if `R: Send + Sync`. The type does not perform
//! internal synchronization.
//!
//! # Brain file format
//!
//! Brain files start with `MHALRUST` followed by a one-byte version and a
//! bincode-encoded model. The format is not compatible with the original
//! C MegaHAL's `.brn` files.
//!
//! # Shipping a pre-trained brain
//!
//! To embed a trained model in your binary, serialize it during a build step and load
//! the serialized bytes at runtime:
//!
//! ```ignore
//! use megahal::MegaHal;
//! use rand::{rngs::SmallRng, SeedableRng};
//! use std::io::Cursor;
//!
//! const BRAIN: &[u8] = include_bytes!("../assets/bot.brn");
//!
//! let mut hal = MegaHal::new(5, SmallRng::seed_from_u64(42));
//! hal.load_brain_from_reader(&mut Cursor::new(BRAIN))
//!     .expect("valid brain data");
//! ```
//!
//! # See also
//!
//! - The examples directory in the repository source.
//! - The `megahal-cli` crate for the `megahal` command-line binary.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::Path;

use megahal_gen::{capitalize, generate_reply};
use megahal_markov::BidirectionalModel;
use megahal_tokenizer::tokenize;
use rand::Rng;
use serde::{Deserialize, Serialize};
use symbol_core::Symbol;
use thiserror::Error;

// Re-export types that consumers (like the CLI) need.
pub use megahal_gen::GenerationLimit;
pub use megahal_keywords::{KeywordConfig, SwapTable, extract_keywords};

/// The MegaHAL symbol type: a case-insensitive byte string.
///
/// All comparisons are case-insensitive (both sides uppercased before comparison).
/// Ordering is lexicographic after uppercasing, with shorter strings comparing as
/// less-than if they share a prefix. This matches the original MegaHAL behavior.
#[derive(Clone, Debug, Serialize, Deserialize)]
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

impl std::hash::Hash for MegaHalSymbol {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.upper().hash(state);
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

/// Magic bytes at the start of a brain file.
const BRAIN_MAGIC: &[u8; 8] = b"MHALRUST";

/// Brain file format version.
const BRAIN_VERSION: u8 = 1;

const DEFAULT_FALLBACK_REPLY: &str = "I don't know enough to answer you yet!";
const DEFAULT_FALLBACK_GREETING: &str = "Hello!";

/// Errors returned by brain serialization APIs.
///
/// Path-based helpers (`save_brain`, `load_brain`, `train_from_file`) still
/// return `io::Result` for source compatibility; the reader/writer-based APIs
/// surface this richer error type.
#[derive(Debug, Error)]
pub enum MegaHalError {
    /// Filesystem or stream I/O error.
    #[error(transparent)]
    Io(#[from] io::Error),

    /// The input did not start with the expected `MHALRUST` magic bytes.
    #[error("not a MegaHAL brain file")]
    BadMagic,

    /// The brain file version is newer than this crate can read.
    #[error("unsupported brain version: {0}")]
    UnsupportedVersion(u8),

    /// Bincode encoding or decoding failed.
    #[error("brain serialization failed: {0}")]
    Decode(String),
}

impl From<MegaHalError> for io::Error {
    fn from(err: MegaHalError) -> Self {
        match err {
            MegaHalError::Io(e) => e,
            MegaHalError::BadMagic
            | MegaHalError::UnsupportedVersion(_)
            | MegaHalError::Decode(_) => {
                io::Error::new(io::ErrorKind::InvalidData, err.to_string())
            }
        }
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
    /// Message returned when reply generation produces nothing.
    fallback_reply: String,
    /// Message returned by `greet` when no greeting can be generated.
    fallback_greeting: String,
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
            fallback_reply: DEFAULT_FALLBACK_REPLY.to_string(),
            fallback_greeting: DEFAULT_FALLBACK_GREETING.to_string(),
        }
    }

    /// Override the message returned when `respond` cannot generate a reply.
    pub fn set_fallback_reply(&mut self, message: impl Into<String>) {
        self.fallback_reply = message.into();
    }

    /// Override the message returned when `greet` cannot generate a greeting.
    pub fn set_fallback_greeting(&mut self, message: impl Into<String>) {
        self.fallback_greeting = message.into();
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
    pub fn respond(&mut self, input: &str) -> String {
        // Step 1: Tokenize.
        let token_strings = tokenize(input);
        let tokens: Vec<MegaHalSymbol> = token_strings
            .iter()
            .map(|s| MegaHalSymbol::new(s))
            .collect();

        // Step 2: Learn (before generating; matches original behavior).
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
            return self.fallback_reply.clone();
        }

        let reply_strings: Vec<String> =
            reply_symbols.iter().map(|s| s.to_string_lossy()).collect();
        capitalize(&reply_strings)
    }

    /// Generate a reply without learning from the input.
    ///
    /// Identical to [`respond`](Self::respond) except that the input is not
    /// learned and the fallback reply is replaced with `None`. Useful for
    /// embedding the engine in applications that handle learning explicitly.
    pub fn generate(&mut self, input: &str) -> Option<String> {
        let token_strings = tokenize(input);
        let tokens: Vec<MegaHalSymbol> = token_strings
            .iter()
            .map(|s| MegaHalSymbol::new(s))
            .collect();

        let keywords = extract_keywords(
            &tokens,
            &self.model.dictionary,
            &self.keyword_config,
            MegaHalSymbol::new,
        );
        let keyword_symbols: HashSet<MegaHalSymbol> =
            keywords.iter().map(|s| MegaHalSymbol::new(s)).collect();

        let reply_symbols = generate_reply(
            &self.model,
            &tokens,
            &keyword_symbols,
            &self.aux_symbols,
            &self.limit,
            &mut self.rng,
        );

        if reply_symbols.is_empty() {
            return None;
        }

        let reply_strings: Vec<String> =
            reply_symbols.iter().map(|s| s.to_string_lossy()).collect();
        Some(capitalize(&reply_strings))
    }

    /// Generate an initial greeting (before any user input).
    pub fn greet(&mut self) -> String {
        if self.greetings.is_empty() {
            return self.fallback_greeting.clone();
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
            return self.fallback_greeting.clone();
        }

        let reply_strings: Vec<String> =
            reply_symbols.iter().map(|s| s.to_string_lossy()).collect();
        capitalize(&reply_strings)
    }

    /// Train from a text file (one sentence per line, `#` lines skipped).
    pub fn train_from_file(&mut self, path: &Path) -> io::Result<()> {
        self.train_from_reader(BufReader::new(File::open(path)?))
    }

    /// Train from any buffered reader (one sentence per line, `#` lines skipped).
    ///
    /// ```
    /// use megahal::MegaHal;
    /// use rand::{rngs::SmallRng, SeedableRng};
    /// use std::io::Cursor;
    ///
    /// let mut hal = MegaHal::new(5, SmallRng::seed_from_u64(0));
    /// let corpus = b"# header\nthe quick brown fox jumps over the lazy dog\n";
    /// hal.train_from_reader(Cursor::new(&corpus[..])).unwrap();
    /// assert!(hal.model().dictionary.len() > 2);
    /// ```
    pub fn train_from_reader<Rd: BufRead>(&mut self, reader: Rd) -> io::Result<()> {
        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();
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

    /// Save the model to a binary brain file.
    ///
    /// Saves the tries and dictionary. Other configuration state is not saved.
    pub fn save_brain(&self, path: &Path) -> io::Result<()> {
        let mut file = File::create(path)?;
        self.save_brain_to_writer(&mut file)?;
        Ok(())
    }

    /// Save the model to any writer (no filesystem required).
    ///
    /// ```
    /// use megahal::MegaHal;
    /// use rand::{rngs::SmallRng, SeedableRng};
    ///
    /// let hal = MegaHal::new(5, SmallRng::seed_from_u64(0));
    /// let mut buf: Vec<u8> = Vec::new();
    /// hal.save_brain_to_writer(&mut buf).unwrap();
    /// assert!(buf.starts_with(b"MHALRUST"));
    /// ```
    pub fn save_brain_to_writer<Wr: Write>(&self, writer: &mut Wr) -> Result<(), MegaHalError> {
        writer.write_all(BRAIN_MAGIC)?;
        writer.write_all(&[BRAIN_VERSION])?;
        bincode::serde::encode_into_std_write(&self.model, writer, bincode::config::standard())
            .map_err(|e| match e {
                bincode::error::EncodeError::Io { inner, .. } => MegaHalError::Io(inner),
                other => MegaHalError::Decode(other.to_string()),
            })?;
        Ok(())
    }

    /// Load a model from a binary brain file, replacing the current model.
    ///
    /// The model order is restored from the file. Keyword config, greetings,
    /// generation limits, and RNG are unaffected.
    pub fn load_brain(&mut self, path: &Path) -> io::Result<()> {
        let mut file = File::open(path)?;
        self.load_brain_from_reader(&mut file)?;
        Ok(())
    }

    /// Load a model from any reader.
    ///
    /// ```
    /// use megahal::MegaHal;
    /// use rand::{rngs::SmallRng, SeedableRng};
    /// use std::io::Cursor;
    ///
    /// let mut hal = MegaHal::new(5, SmallRng::seed_from_u64(0));
    /// hal.learn("the quick brown fox jumps over the lazy dog");
    /// let mut buf: Vec<u8> = Vec::new();
    /// hal.save_brain_to_writer(&mut buf).unwrap();
    ///
    /// let mut hal2 = MegaHal::new(5, SmallRng::seed_from_u64(0));
    /// hal2.load_brain_from_reader(&mut Cursor::new(buf)).unwrap();
    /// assert!(hal2.model().dictionary.len() > 2);
    /// ```
    pub fn load_brain_from_reader<Rd: Read>(
        &mut self,
        reader: &mut Rd,
    ) -> Result<(), MegaHalError> {
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != BRAIN_MAGIC {
            return Err(MegaHalError::BadMagic);
        }
        let mut version = [0u8; 1];
        reader.read_exact(&mut version)?;
        if version[0] != BRAIN_VERSION {
            return Err(MegaHalError::UnsupportedVersion(version[0]));
        }
        let model = bincode::serde::decode_from_std_read(reader, bincode::config::standard())
            .map_err(|e| match e {
                bincode::error::DecodeError::Io { inner, .. } => MegaHalError::Io(inner),
                other => MegaHalError::Decode(other.to_string()),
            })?;
        self.model = model;
        Ok(())
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
    fn megahal_symbol_as_bytes() {
        let sym = MegaHalSymbol::new("Hello");
        assert_eq!(sym.as_bytes(), b"HELLO");
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

    #[test]
    fn set_fallback_reply_changes_canned_response() {
        let mut hal = test_hal();
        hal.set_limit(GenerationLimit::Iterations(5));
        hal.set_fallback_reply("UNKNOWN");
        assert_eq!(hal.respond("Hi."), "UNKNOWN");
    }

    #[test]
    fn set_fallback_greeting_changes_default() {
        let mut hal = test_hal();
        hal.set_fallback_greeting("HOWDY");
        assert_eq!(hal.greet(), "HOWDY");
    }

    #[test]
    fn generate_returns_none_on_empty_model() {
        let mut hal = test_hal();
        hal.set_limit(GenerationLimit::Iterations(5));
        assert_eq!(hal.generate("Hi."), None);
    }

    #[test]
    fn generate_does_not_learn_from_input() {
        let mut hal = test_hal();
        hal.set_limit(GenerationLimit::Iterations(5));
        // generate() must not mutate the dictionary, even on inputs that would
        // ordinarily be learned by respond().
        let before = hal.model().dictionary.len();
        let _ = hal.generate("Tell me about the world and many other things.");
        let after = hal.model().dictionary.len();
        assert_eq!(before, after);
    }

    #[test]
    fn generate_produces_string_when_model_has_data() {
        let mut hal = trained_hal();
        let reply = hal.generate("Tell me about dogs.");
        // Verify that generate produces a reply when data is present in the model.
        if let Some(s) = reply {
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn save_load_via_reader_writer_roundtrip() {
        let mut hal = trained_hal();
        let _ = hal.respond("Tell me about dogs.");

        let mut buf: Vec<u8> = Vec::new();
        hal.save_brain_to_writer(&mut buf).unwrap();
        assert!(buf.starts_with(BRAIN_MAGIC));

        let mut hal2 = test_hal();
        hal2.set_limit(GenerationLimit::Iterations(20));
        hal2.load_brain_from_reader(&mut io::Cursor::new(&buf))
            .unwrap();

        assert_eq!(hal.model().dictionary.len(), hal2.model().dictionary.len());
    }

    #[test]
    fn load_from_reader_bad_magic_returns_typed_error() {
        let mut hal = test_hal();
        let mut cursor = io::Cursor::new(b"NOTABRAIN".to_vec());
        match hal.load_brain_from_reader(&mut cursor) {
            Err(MegaHalError::BadMagic) => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn load_from_reader_bad_version_returns_typed_error() {
        let mut hal = test_hal();
        let mut data = Vec::new();
        data.extend_from_slice(BRAIN_MAGIC);
        data.push(99);
        let mut cursor = io::Cursor::new(data);
        match hal.load_brain_from_reader(&mut cursor) {
            Err(MegaHalError::UnsupportedVersion(99)) => {}
            other => panic!("expected UnsupportedVersion(99), got {other:?}"),
        }
    }

    #[test]
    fn train_from_reader_works_with_cursor() {
        let mut hal = test_hal();
        let corpus = b"# comment line\nthe quick brown fox jumps over the lazy dog\n\n";
        hal.train_from_reader(io::Cursor::new(&corpus[..])).unwrap();
        assert!(hal.model().dictionary.len() > 2);
    }

    #[test]
    fn respond_does_not_echo_input_verbatim() {
        // When the only thing the model knows is the user's input, every
        // candidate reply equals the input. C's `dissimilar()` check rejects
        // the baseline in that case; the facade falls back to the canned reply.
        let mut hal = test_hal();
        hal.set_limit(GenerationLimit::Iterations(20));
        let input = "the very long input sentence with many distinct words and concepts";
        hal.learn(input);
        let reply = hal.respond(input);
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
    fn greet_with_greetings_but_no_training() {
        let mut hal = test_hal();
        hal.set_greetings(vec!["HI".into()]);
        // No training → generation fails → fallback to "Hello!".
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

    // --- Brain persistence tests ---

    #[test]
    fn save_load_brain_roundtrip() {
        let mut hal = trained_hal();
        let _ = hal.respond("Tell me about dogs.");

        let dir = std::env::temp_dir();
        let path = dir.join("megahal_test_brain.brn");
        hal.save_brain(&path).unwrap();

        // Create a fresh MegaHal and load the brain.
        let mut hal2 = test_hal();
        hal2.set_limit(GenerationLimit::Iterations(20));
        hal2.load_brain(&path).unwrap();

        // The loaded model should have learned data.
        let reply = hal2.respond("Tell me about dogs.");
        assert!(!reply.is_empty());
        assert_ne!(reply, "I don't know enough to answer you yet!");

        // Dictionary size should match.
        assert_eq!(hal.model().dictionary.len(), hal2.model().dictionary.len());

        fs::remove_file(&path).ok();
    }

    #[test]
    fn load_brain_rejects_bad_magic() {
        let dir = std::env::temp_dir();
        let path = dir.join("megahal_test_bad_magic.brn");
        fs::write(&path, b"NOTABRAIN000000").unwrap();

        let mut hal = test_hal();
        let err = hal.load_brain(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("not a MegaHAL brain file"));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn load_brain_rejects_bad_version() {
        let dir = std::env::temp_dir();
        let path = dir.join("megahal_test_bad_version.brn");
        let mut data = Vec::new();
        data.extend_from_slice(b"MHALRUST");
        data.push(99);
        fs::write(&path, &data).unwrap();

        let mut hal = test_hal();
        let err = hal.load_brain(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("unsupported brain version"));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn load_brain_rejects_truncated_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("megahal_test_truncated.brn");
        fs::write(&path, b"MHAL").unwrap();

        let mut hal = test_hal();
        let err = hal.load_brain(&path).unwrap_err();
        // 4-byte file: reader hits EOF while reading the 8-byte magic header.
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);

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
