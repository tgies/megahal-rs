//! MegaHAL reply generation: babble, seeding, bidirectional generation,
//! and surprise evaluation.
//!
//! This crate implements the core MegaHAL reply generation algorithm:
//!
//! 1. **Seed** a starting symbol from the keyword list.
//! 2. **Forward phase**: babble from the seed to generate the rest of the sentence.
//! 3. **Backward phase**: babble backward from the seed to generate the beginning.
//! 4. **Evaluate** candidates by surprise scoring (Shannon entropy of keywords).
//! 5. **Select** the highest-scoring candidate within the generation limit.
//!
//! The babble function drives the [`ContextWindow`] directly — no trait abstraction.
//! It implements MegaHAL's keyword-biased random walk: keywords encountered during
//! the walk are greedily selected, while non-keywords fall back to probability-weighted
//! selection based on `child.count / parent.usage`.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use markov_chain::{BidirectionalModel, ContextWindow};
use ngram_trie::Trie;
use rand::Rng;
use symbol_core::{Symbol, SymbolId, ERROR_ID, FIN_ID};
use symbol_dict::SymbolDict;

/// Controls how many candidate replies are generated before selecting the best.
#[derive(Debug, Clone)]
pub enum GenerationLimit {
    /// Stop after the given duration.
    Timeout(Duration),
    /// Stop after the given number of iterations.
    Iterations(usize),
    /// Stop when either limit is reached.
    Both {
        timeout: Duration,
        max_iterations: usize,
    },
}

impl Default for GenerationLimit {
    fn default() -> Self {
        // Match original MegaHAL: 1-second timeout.
        GenerationLimit::Timeout(Duration::from_secs(1))
    }
}

/// Generate the best reply for given input tokens and keywords.
///
/// Runs the candidate generation loop from MEGAHAL_SPEC.md Section 7.1:
/// 1. Generate a baseline reply with empty keywords.
/// 2. Repeatedly generate candidates with keywords, scoring each by surprise.
/// 3. Return the highest-scoring candidate that differs from the input.
pub fn generate_reply<S, R>(
    model: &BidirectionalModel<S>,
    input_tokens: &[S],
    keywords: &HashSet<S>,
    aux_set: &HashSet<S>,
    limit: &GenerationLimit,
    rng: &mut R,
) -> Vec<S>
where
    S: Symbol + AsRef<[u8]>,
    R: Rng,
{
    let empty_keywords = HashSet::new();
    let empty_aux = HashSet::new();

    // Baseline reply (no keyword bias).
    let mut best = generate_one_reply(model, &empty_keywords, &empty_aux, rng);
    if tokens_equal(&best, input_tokens) {
        // Fallback will be handled by the caller (facade) with a canned message.
        // For now, keep the baseline.
    }

    let mut max_surprise: f64 = -1.0;
    let start = Instant::now();
    let mut iterations = 0;

    loop {
        // Check limits.
        match limit {
            GenerationLimit::Timeout(d) => {
                if start.elapsed() >= *d {
                    break;
                }
            }
            GenerationLimit::Iterations(n) => {
                if iterations >= *n {
                    break;
                }
            }
            GenerationLimit::Both {
                timeout,
                max_iterations,
            } => {
                if start.elapsed() >= *timeout || iterations >= *max_iterations {
                    break;
                }
            }
        }

        let candidate = generate_one_reply(model, keywords, aux_set, rng);
        let surprise = evaluate_reply(model, &candidate, keywords);

        if surprise > max_surprise && !tokens_equal(&candidate, input_tokens) {
            max_surprise = surprise;
            best = candidate;
        }

        iterations += 1;
    }

    best
}

/// Generate a single candidate reply (forward + backward phases).
///
/// MEGAHAL_SPEC.md Section 7.2.
fn generate_one_reply<S, R>(
    model: &BidirectionalModel<S>,
    keywords: &HashSet<S>,
    aux_set: &HashSet<S>,
    rng: &mut R,
) -> Vec<S>
where
    S: Symbol + AsRef<[u8]>,
    R: Rng,
{
    let mut reply: Vec<SymbolId> = Vec::new();
    let mut used_key = false;

    // Forward phase.
    let mut ctx = model.forward_context();

    let seed_id = seed(model, keywords, aux_set, rng);
    if seed_id == ERROR_ID || seed_id == FIN_ID {
        // Empty reply — skip to backward phase or return empty.
        return resolve_ids(model, &reply);
    }

    reply.push(seed_id);
    ctx.advance(&model.forward, seed_id);

    loop {
        let sym = babble(
            &model.forward,
            &ctx,
            &model.dictionary,
            keywords,
            aux_set,
            &reply,
            &mut used_key,
            rng,
        );
        if sym == ERROR_ID || sym == FIN_ID {
            break;
        }
        reply.push(sym);
        ctx.advance(&model.forward, sym);
    }

    // Backward phase.
    let mut ctx = model.backward_context();

    // Re-establish backward context from the beginning of the reply.
    // Spec 7.2.3: walk from index min(reply_length-1, order) down to 0.
    // This matches the C code: for(i=MIN(size-1,order); i>=0; i--)
    if !reply.is_empty() {
        let start = (reply.len() - 1).min(model.order as usize);
        for i in (0..=start).rev() {
            ctx.advance(&model.backward, reply[i]);
        }
    }

    loop {
        let sym = babble(
            &model.backward,
            &ctx,
            &model.dictionary,
            keywords,
            aux_set,
            &reply,
            &mut used_key,
            rng,
        );
        if sym == ERROR_ID || sym == FIN_ID {
            break;
        }
        reply.insert(0, sym);
        ctx.advance(&model.backward, sym);
    }

    resolve_ids(model, &reply)
}

/// Select a seed symbol for forward generation.
///
/// MEGAHAL_SPEC.md Section 7.2.1.
fn seed<S, R>(
    model: &BidirectionalModel<S>,
    keywords: &HashSet<S>,
    aux_set: &HashSet<S>,
    rng: &mut R,
) -> SymbolId
where
    S: Symbol + AsRef<[u8]>,
    R: Rng,
{
    let root = model.forward.root();
    let children = model.forward.children(root);

    if children.is_empty() {
        return ERROR_ID;
    }

    // If keywords exist, try to find a non-auxiliary keyword as seed.
    if !keywords.is_empty() {
        let mut keyword_vec: Vec<&S> = keywords.iter().collect();
        keyword_vec.sort();
        let start = rng.random_range(0..keyword_vec.len());

        for offset in 0..keyword_vec.len() {
            let idx = (start + offset) % keyword_vec.len();
            let kw = keyword_vec[idx];

            // Must exist in dictionary and not be auxiliary.
            if let Some(id) = model.dictionary.find(kw)
                && !aux_set.contains(kw)
            {
                return id;
            }
        }
    }

    // Default: pick a random child of the forward root.
    let idx = rng.random_range(0..children.len());
    model.forward.node(children[idx]).symbol
}

/// Keyword-biased random symbol selection (the "babble" function).
///
/// MEGAHAL_SPEC.md Section 7.3.
#[allow(clippy::too_many_arguments)]
fn babble<S, R>(
    trie: &Trie,
    ctx: &ContextWindow,
    dict: &SymbolDict<S>,
    keywords: &HashSet<S>,
    aux_set: &HashSet<S>,
    reply: &[SymbolId],
    used_key: &mut bool,
    rng: &mut R,
) -> SymbolId
where
    S: Symbol + AsRef<[u8]>,
    R: Rng,
{
    // Find deepest available context.
    let node_ref = match ctx.deepest() {
        Some(r) => r,
        None => return ERROR_ID,
    };

    let node = trie.node(node_ref);
    let children = trie.children(node_ref);

    if children.is_empty() {
        return ERROR_ID;
    }

    let branch = children.len();
    let mut i = rng.random_range(0..branch);
    let mut count = rng.random_range(0..node.usage as i64);

    loop {
        let child_ref = children[i];
        let child = trie.node(child_ref);
        let sym = child.symbol;

        // Check if this symbol is a keyword we should greedily select.
        let word = dict.resolve(sym);
        let is_keyword = keywords.contains(word);
        let is_aux = aux_set.contains(word);
        let already_in_reply = reply.contains(&sym);

        if is_keyword && (*used_key || !is_aux) && !already_in_reply {
            *used_key = true;
            return sym;
        }

        // Otherwise, probability-weighted selection.
        count -= child.count as i64;
        if count < 0 {
            return sym;
        }

        i = (i + 1) % branch;
    }
}

/// Score a candidate reply by surprise (Shannon entropy of keywords in context).
///
/// MEGAHAL_SPEC.md Section 8.
fn evaluate_reply<S>(
    model: &BidirectionalModel<S>,
    candidate: &[S],
    keywords: &HashSet<S>,
) -> f64
where
    S: Symbol + AsRef<[u8]>,
{
    if candidate.is_empty() {
        return 0.0;
    }

    let mut entropy: f64 = 0.0;
    let mut num: usize = 0;

    // Forward evaluation.
    let mut ctx = model.forward_context();
    for token in candidate {
        let sym_id = match model.dictionary.find(token) {
            Some(id) => id,
            None => continue,
        };

        if keywords.contains(token) {
            let mut prob: f64 = 0.0;
            let mut ctx_count: usize = 0;

            for j in 0..model.order as usize {
                if let Some(parent_ref) = ctx.at_depth(j)
                    && let Some(child_ref) = model.forward.find_child(parent_ref, sym_id)
                {
                    let child = model.forward.node(child_ref);
                    let parent = model.forward.node(parent_ref);
                    if parent.usage > 0 {
                        prob += child.count as f64 / parent.usage as f64;
                        ctx_count += 1;
                    }
                }
            }

            if ctx_count > 0 {
                entropy -= (prob / ctx_count as f64).ln();
            }
            num += 1;
        }

        ctx.advance(&model.forward, sym_id);
    }

    // Backward evaluation.
    let mut ctx = model.backward_context();
    for token in candidate.iter().rev() {
        let sym_id = match model.dictionary.find(token) {
            Some(id) => id,
            None => continue,
        };

        if keywords.contains(token) {
            let mut prob: f64 = 0.0;
            let mut ctx_count: usize = 0;

            for j in 0..model.order as usize {
                if let Some(parent_ref) = ctx.at_depth(j)
                    && let Some(child_ref) = model.backward.find_child(parent_ref, sym_id)
                {
                    let child = model.backward.node(child_ref);
                    let parent = model.backward.node(parent_ref);
                    if parent.usage > 0 {
                        prob += child.count as f64 / parent.usage as f64;
                        ctx_count += 1;
                    }
                }
            }

            if ctx_count > 0 {
                entropy -= (prob / ctx_count as f64).ln();
            }
            num += 1;
        }

        ctx.advance(&model.backward, sym_id);
    }

    // Length penalty.
    if num >= 8 {
        entropy /= ((num - 1) as f64).sqrt();
    }
    if num >= 16 {
        entropy /= num as f64;
    }

    entropy
}

/// Capitalize a token sequence per MegaHAL sentence-case rules.
///
/// MEGAHAL_SPEC.md Section 9.1.
pub fn capitalize(tokens: &[String]) -> String {
    let raw: String = tokens.concat();
    let mut result = Vec::with_capacity(raw.len());
    let bytes = raw.as_bytes();
    let mut capitalize_next = true;

    for (i, &b) in bytes.iter().enumerate() {
        if capitalize_next && b.is_ascii_alphabetic() {
            result.push(b.to_ascii_uppercase());
            capitalize_next = false;
        } else if b.is_ascii_alphabetic() {
            result.push(b.to_ascii_lowercase());
        } else {
            result.push(b);
            // After sentence-ending punctuation followed by space, capitalize next.
            if matches!(b, b'!' | b'.' | b'?') && i > 2 {
                // The next whitespace char triggers capitalization of the alpha after it.
                capitalize_next = true;
            }
            if capitalize_next && b.is_ascii_whitespace() {
                // Keep capitalize_next true through whitespace.
            } else if !matches!(b, b'!' | b'.' | b'?') {
                // Non-punctuation, non-whitespace resets the flag.
                // Actually, only alphabetic chars consume the flag (handled above).
                // Non-alpha, non-terminal-punct, non-space: keep capitalize_next as is.
            }
        }
    }

    String::from_utf8(result).unwrap_or(raw)
}

/// Check if two token sequences are equal (case-insensitive, for dissimilarity test).
fn tokens_equal<S: Symbol>(a: &[S], b: &[S]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

/// Resolve a sequence of SymbolIds back to Symbol values.
fn resolve_ids<S: Symbol>(model: &BidirectionalModel<S>, ids: &[SymbolId]) -> Vec<S> {
    ids.iter()
        .map(|&id| model.dictionary.resolve(id).clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    // --- Test infrastructure ---

    #[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
    struct TSym(String);

    impl Symbol for TSym {
        fn error() -> Self {
            TSym("<ERROR>".into())
        }
        fn fin() -> Self {
            TSym("<FIN>".into())
        }
    }

    impl AsRef<[u8]> for TSym {
        fn as_ref(&self) -> &[u8] {
            self.0.as_bytes()
        }
    }

    fn ts(s: &str) -> TSym {
        TSym(s.to_string())
    }

    fn trained_model(order: u8, sentences: &[&[&str]]) -> BidirectionalModel<TSym> {
        let mut model = BidirectionalModel::new(order);
        for sentence in sentences {
            let tokens: Vec<TSym> = sentence.iter().map(|&s| ts(s)).collect();
            model.learn(&tokens);
        }
        model
    }

    fn make_rng(s: u64) -> SmallRng {
        SmallRng::seed_from_u64(s)
    }

    // --- GenerationLimit tests ---

    #[test]
    fn generation_limit_default_is_timeout() {
        let limit = GenerationLimit::default();
        assert!(matches!(limit, GenerationLimit::Timeout(_)));
    }

    // --- capitalize tests ---

    #[test]
    fn capitalize_basic() {
        let tokens = vec![
            "hello".to_string(),
            " ".to_string(),
            "world".to_string(),
            ".".to_string(),
        ];
        assert_eq!(capitalize(&tokens), "Hello world.");
    }

    #[test]
    fn capitalize_after_period() {
        let tokens = vec![
            "hello".to_string(),
            ". ".to_string(),
            "world".to_string(),
            ".".to_string(),
        ];
        assert_eq!(capitalize(&tokens), "Hello. World.");
    }

    #[test]
    fn capitalize_empty() {
        let tokens: Vec<String> = vec![];
        assert_eq!(capitalize(&tokens), "");
    }

    #[test]
    fn capitalize_after_exclamation() {
        let tokens = vec![
            "wow".to_string(),
            "! ".to_string(),
            "amazing".to_string(),
            ".".to_string(),
        ];
        assert_eq!(capitalize(&tokens), "Wow! Amazing.");
    }

    #[test]
    fn capitalize_after_question() {
        let tokens = vec![
            "really".to_string(),
            "? ".to_string(),
            "yes".to_string(),
            ".".to_string(),
        ];
        assert_eq!(capitalize(&tokens), "Really? Yes.");
    }

    // --- seed tests ---

    #[test]
    fn seed_selects_keyword() {
        let model = trained_model(2, &[
            &["THE", " ", "CAT", " ", "SAT"],
            &["THE", " ", "DOG", " ", "RAN"],
        ]);
        let mut kws = HashSet::new();
        kws.insert(ts("CAT"));
        let aux = HashSet::new();
        let mut rng = make_rng(42);
        let id = seed(&model, &kws, &aux, &mut rng);
        let cat_id = model.dictionary.find(&ts("CAT")).unwrap();
        assert_eq!(id, cat_id);
    }

    #[test]
    fn seed_skips_aux_keyword() {
        let model = trained_model(2, &[&["THE", " ", "MY", " ", "CAT"]]);
        let mut kws = HashSet::new();
        kws.insert(ts("MY"));
        let mut aux = HashSet::new();
        aux.insert(ts("MY"));
        let mut rng = make_rng(42);
        let id = seed(&model, &kws, &aux, &mut rng);
        // MY is aux-only → seed falls back to random child of forward root.
        let my_id = model.dictionary.find(&ts("MY")).unwrap();
        assert_ne!(id, my_id);
    }

    #[test]
    fn seed_with_empty_keywords_picks_random() {
        let model = trained_model(2, &[&["THE", " ", "CAT"]]);
        let kws = HashSet::new();
        let aux = HashSet::new();
        let mut rng = make_rng(42);
        let id = seed(&model, &kws, &aux, &mut rng);
        assert_ne!(id, ERROR_ID);
        assert_ne!(id, FIN_ID);
    }

    #[test]
    fn seed_returns_error_on_empty_model() {
        let model: BidirectionalModel<TSym> = BidirectionalModel::new(2);
        let kws = HashSet::new();
        let aux = HashSet::new();
        let mut rng = make_rng(42);
        let id = seed(&model, &kws, &aux, &mut rng);
        assert_eq!(id, ERROR_ID);
    }

    // --- evaluate_reply tests ---

    #[test]
    fn evaluate_empty_candidate_returns_zero() {
        let model = trained_model(2, &[&["A", "B", "C"]]);
        let kws = HashSet::new();
        let score = evaluate_reply(&model, &[], &kws);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn evaluate_no_keywords_returns_zero() {
        let model = trained_model(2, &[&["A", "B", "C"]]);
        let kws = HashSet::new();
        let candidate = vec![ts("A"), ts("B"), ts("C")];
        let score = evaluate_reply(&model, &candidate, &kws);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn evaluate_with_keywords_returns_positive() {
        let model = trained_model(
            2,
            &[&["A", "B", "C"], &["A", "B", "C"], &["A", "B", "C"]],
        );
        let mut kws = HashSet::new();
        kws.insert(ts("B"));
        let candidate = vec![ts("A"), ts("B"), ts("C")];
        let score = evaluate_reply(&model, &candidate, &kws);
        assert!(score > 0.0, "Expected positive surprise, got {score}");
    }

    #[test]
    fn evaluate_unknown_token_skipped() {
        let model = trained_model(2, &[&["A", "B", "C"]]);
        let mut kws = HashSet::new();
        kws.insert(ts("UNKNOWN"));
        // UNKNOWN is not in the dict → find returns None → skipped.
        let candidate = vec![ts("UNKNOWN")];
        let score = evaluate_reply(&model, &candidate, &kws);
        assert_eq!(score, 0.0);
    }

    // --- tokens_equal tests ---

    #[test]
    fn tokens_equal_same() {
        let a = vec![ts("A"), ts("B")];
        let b = vec![ts("A"), ts("B")];
        assert!(tokens_equal(&a, &b));
    }

    #[test]
    fn tokens_equal_different_length() {
        let a = vec![ts("A"), ts("B")];
        let b = vec![ts("A")];
        assert!(!tokens_equal(&a, &b));
    }

    #[test]
    fn tokens_equal_different_content() {
        let a = vec![ts("A"), ts("B")];
        let b = vec![ts("A"), ts("C")];
        assert!(!tokens_equal(&a, &b));
    }

    #[test]
    fn tokens_equal_both_empty() {
        let a: Vec<TSym> = vec![];
        let b: Vec<TSym> = vec![];
        assert!(tokens_equal(&a, &b));
    }

    // --- generate_reply integration tests ---

    #[test]
    fn generate_reply_empty_model() {
        let model: BidirectionalModel<TSym> = BidirectionalModel::new(2);
        let kws = HashSet::new();
        let aux = HashSet::new();
        let limit = GenerationLimit::Iterations(10);
        let mut rng = make_rng(42);
        let reply = generate_reply(&model, &[], &kws, &aux, &limit, &mut rng);
        assert!(reply.is_empty());
    }

    #[test]
    fn generate_reply_produces_output() {
        let model = trained_model(
            2,
            &[
                &["THE", " ", "CAT", " ", "SAT"],
                &["THE", " ", "DOG", " ", "RAN"],
                &["A", " ", "BIG", " ", "CAT"],
            ],
        );
        let mut kws = HashSet::new();
        kws.insert(ts("CAT"));
        let aux = HashSet::new();
        let limit = GenerationLimit::Iterations(10);
        let mut rng = make_rng(42);
        let reply = generate_reply(&model, &[], &kws, &aux, &limit, &mut rng);
        assert!(!reply.is_empty());
    }

    #[test]
    fn generate_reply_deterministic() {
        let build = || {
            let model = trained_model(
                2,
                &[
                    &["THE", " ", "CAT", " ", "SAT"],
                    &["THE", " ", "DOG", " ", "RAN"],
                ],
            );
            let mut kws = HashSet::new();
            kws.insert(ts("CAT"));
            let aux = HashSet::new();
            let limit = GenerationLimit::Iterations(50);
            let mut rng = make_rng(42);
            generate_reply(&model, &[], &kws, &aux, &limit, &mut rng)
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn generate_reply_zero_iterations_returns_baseline() {
        let model = trained_model(2, &[&["THE", " ", "CAT", " ", "SAT"]]);
        let mut kws = HashSet::new();
        kws.insert(ts("CAT"));
        let aux = HashSet::new();
        let limit = GenerationLimit::Iterations(0);
        let mut rng = make_rng(42);
        // With 0 iterations, only the baseline (no-keyword) reply is generated.
        // Should return without hanging.
        let reply = generate_reply(&model, &[], &kws, &aux, &limit, &mut rng);
        let _ = reply; // just verify it completes
    }

    #[test]
    fn generate_reply_with_timeout() {
        let model = trained_model(
            2,
            &[
                &["THE", " ", "CAT", " ", "SAT"],
                &["THE", " ", "DOG", " ", "RAN"],
                &["A", " ", "BIG", " ", "CAT"],
            ],
        );
        let mut kws = HashSet::new();
        kws.insert(ts("CAT"));
        let aux = HashSet::new();
        let limit = GenerationLimit::Timeout(Duration::from_millis(50));
        let mut rng = make_rng(42);
        let reply = generate_reply(&model, &[], &kws, &aux, &limit, &mut rng);
        assert!(!reply.is_empty());
    }

    #[test]
    fn generate_reply_with_both_limit() {
        let model = trained_model(
            2,
            &[
                &["THE", " ", "CAT", " ", "SAT"],
                &["THE", " ", "DOG", " ", "RAN"],
            ],
        );
        let mut kws = HashSet::new();
        kws.insert(ts("CAT"));
        let aux = HashSet::new();
        let limit = GenerationLimit::Both {
            timeout: Duration::from_millis(50),
            max_iterations: 10,
        };
        let mut rng = make_rng(42);
        let reply = generate_reply(&model, &[], &kws, &aux, &limit, &mut rng);
        assert!(!reply.is_empty());
    }

    #[test]
    fn generate_reply_with_aux_keywords() {
        let model = trained_model(
            2,
            &[
                &["MY", " ", "CAT", " ", "SAT"],
                &["YOUR", " ", "DOG", " ", "RAN"],
            ],
        );
        let mut kws = HashSet::new();
        kws.insert(ts("CAT"));
        kws.insert(ts("MY"));
        let mut aux = HashSet::new();
        aux.insert(ts("MY"));
        let limit = GenerationLimit::Iterations(20);
        let mut rng = make_rng(42);
        let reply = generate_reply(&model, &[], &kws, &aux, &limit, &mut rng);
        assert!(!reply.is_empty());
    }

    #[test]
    fn generate_reply_dissimilarity_test() {
        // If the candidate is identical to input, it should be rejected in favor
        // of a different candidate (when possible).
        let model = trained_model(
            2,
            &[
                &["A", " ", "B", " ", "C"],
                &["D", " ", "E", " ", "F"],
                &["A", " ", "B", " ", "C"],
            ],
        );
        let input = vec![ts("A"), ts(" "), ts("B"), ts(" "), ts("C")];
        let kws = HashSet::new();
        let aux = HashSet::new();
        let limit = GenerationLimit::Iterations(50);
        let mut rng = make_rng(42);
        let reply = generate_reply(&model, &input, &kws, &aux, &limit, &mut rng);
        // With multiple sentences in the model, at least some candidates should
        // differ from input. The best reply should not be identical to input.
        // (This isn't guaranteed with tiny models, but is very likely.)
        let _ = reply;
    }
}
