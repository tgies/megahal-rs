//! MegaHAL keyword extraction: two-pass algorithm with swap table,
//! banned/auxiliary word lists.
//!
//! Keywords drive MegaHAL's reply generation by biasing the Markov walk toward
//! topically relevant symbols. Extraction works in two passes:
//!
//! 1. **Primary**: select words from input (after swap substitution) that are
//!    in the model dictionary, start with an alphanumeric character, and are
//!    neither banned nor auxiliary.
//! 2. **Auxiliary**: if at least one primary keyword was found, also add words
//!    from the auxiliary list (pronouns, possessives) that appear in input.

use std::collections::HashSet;

use symbol_core::Symbol;
use symbol_dict::SymbolDict;

/// Perspective-swapping substitution table.
///
/// When extracting keywords, input tokens are matched against `from` entries.
/// If a match is found, the corresponding `to` entry is used as the keyword
/// candidate instead. Multiple `from` entries can match the same token,
/// producing multiple keyword candidates (e.g., "YOU" → ["I", "ME"]).
#[derive(Debug, Clone, Default)]
pub struct SwapTable {
    /// (from, to) pairs. Scanned linearly for each input token.
    pub pairs: Vec<(String, String)>,
}

impl SwapTable {
    /// Apply swap substitutions to a token. Returns all matching `to` values.
    /// If no match, returns the original token.
    pub fn apply(&self, token: &str) -> Vec<String> {
        let mut results = Vec::new();
        for (from, to) in &self.pairs {
            if from.eq_ignore_ascii_case(token) {
                results.push(to.clone());
            }
        }
        if results.is_empty() {
            results.push(token.to_string());
        }
        results
    }
}

/// Configuration for keyword extraction.
#[derive(Debug, Clone, Default)]
pub struct KeywordConfig {
    /// Words that are never used as keywords (common function words).
    pub banned: HashSet<String>,
    /// Words used as keywords only to supplement existing primary keywords.
    pub auxiliary: HashSet<String>,
    /// Perspective-swapping substitutions.
    pub swap: SwapTable,
}

/// Extract keywords from tokenized input per the MegaHAL two-pass algorithm.
///
/// `S` must implement `AsRef<[u8]>` so we can check if the first character is
/// alphanumeric (a requirement of the extraction rules).
///
/// `make_symbol` constructs a Symbol from a string for dictionary lookup. This
/// is needed because after swap substitution, candidates are Strings, but the
/// dictionary is keyed on `S`. The caller (which knows the concrete Symbol type)
/// provides this factory.
///
/// Returns the keyword set as uppercase `String` values (matching the model's
/// internal representation).
pub fn extract_keywords<S: Symbol + AsRef<[u8]>>(
    tokens: &[S],
    dict: &SymbolDict<S>,
    config: &KeywordConfig,
    make_symbol: impl Fn(&str) -> S,
) -> HashSet<String> {
    let mut keywords = HashSet::new();

    // Collect all swap-applied candidates for the two-pass algorithm.
    let candidates: Vec<Vec<String>> = tokens
        .iter()
        .map(|tok| {
            let tok_str = std::str::from_utf8(tok.as_ref()).unwrap_or("");
            config.swap.apply(tok_str)
        })
        .collect();

    // Pass 1: Primary keywords.
    for candidate_group in &candidates {
        for candidate in candidate_group {
            if !is_keyword_eligible(candidate, dict, config, false, &make_symbol) {
                continue;
            }
            keywords.insert(candidate.clone());
        }
    }

    // Pass 2: Auxiliary keywords (only if primary pass found at least one).
    if !keywords.is_empty() {
        for candidate_group in &candidates {
            for candidate in candidate_group {
                if !is_keyword_eligible(candidate, dict, config, true, &make_symbol) {
                    continue;
                }
                keywords.insert(candidate.clone());
            }
        }
    }

    keywords
}

/// Check if a candidate word is eligible as a keyword.
///
/// In primary mode (`aux_pass = false`): must be in dict, start alphanumeric,
/// not banned, not auxiliary.
/// In auxiliary mode (`aux_pass = true`): must be in dict, start alphanumeric,
/// and IS in auxiliary list.
fn is_keyword_eligible<S: Symbol + AsRef<[u8]>>(
    candidate: &str,
    dict: &SymbolDict<S>,
    config: &KeywordConfig,
    aux_pass: bool,
    make_symbol: &impl Fn(&str) -> S,
) -> bool {
    // Must start with an alphanumeric character.
    let first_byte = candidate.as_bytes().first().copied();
    if !first_byte.is_some_and(|b| b.is_ascii_alphanumeric()) {
        return false;
    }

    // Must exist in the model dictionary (the model has seen this word).
    let sym = make_symbol(candidate);
    if dict.find(&sym).is_none() {
        return false;
    }

    let upper = candidate.to_uppercase();

    if aux_pass {
        // Auxiliary pass: only add words that ARE in the auxiliary list.
        config.auxiliary.contains(&upper)
    } else {
        // Primary pass: skip banned and auxiliary words.
        if config.banned.contains(&upper) {
            return false;
        }
        if config.auxiliary.contains(&upper) {
            return false;
        }
        true
    }
}

/// Check if a given string exists in the model dictionary.
///
/// This requires constructing a temporary Symbol, which is the responsibility
/// of the caller (the facade crate knows how to create `MegaHalSymbol` from strings).
pub fn word_in_dict<S: Symbol>(dict: &SymbolDict<S>, symbol: &S) -> bool {
    dict.find(symbol).is_some_and(|id| id != symbol_core::ERROR_ID)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_table_basic() {
        let swap = SwapTable {
            pairs: vec![
                ("I".into(), "YOU".into()),
                ("YOU".into(), "I".into()),
                ("YOU".into(), "ME".into()),
            ],
        };

        assert_eq!(swap.apply("I"), vec!["YOU"]);

        let you_swaps = swap.apply("YOU");
        assert_eq!(you_swaps, vec!["I", "ME"]); // multiple matches

        assert_eq!(swap.apply("HELLO"), vec!["HELLO"]); // no match → original
    }

    #[test]
    fn swap_case_insensitive() {
        let swap = SwapTable {
            pairs: vec![("MY".into(), "YOUR".into())],
        };
        assert_eq!(swap.apply("my"), vec!["YOUR"]);
        assert_eq!(swap.apply("My"), vec!["YOUR"]);
    }

    #[test]
    fn keyword_config_default() {
        let config = KeywordConfig::default();
        assert!(config.banned.is_empty());
        assert!(config.auxiliary.is_empty());
        assert!(config.swap.pairs.is_empty());
    }
}
