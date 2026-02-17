//! MegaHAL text tokenization: boundary detection, apostrophe rules,
//! and sentence-terminal normalization.
//!
//! Splits input text into an alternating sequence of word tokens and separator
//! tokens (whitespace, punctuation). Contractions like "DON'T" and "I'M" are
//! kept as single tokens via the apostrophe rule.
//!
//! This crate has no dependencies on other megahal crates — it is a pure text
//! processing utility that produces `Vec<String>`.

/// Tokenize input text per MegaHAL rules.
///
/// 1. Converts to uppercase.
/// 2. Splits on word boundaries (alpha/digit transitions, with apostrophe
///    exception for contractions).
/// 3. Ensures the token sequence ends with sentence-terminal punctuation
///    (`!`, `.`, or `?`).
///
/// # Examples
///
/// ```
/// use megahal_tokenizer::tokenize;
///
/// let tokens = tokenize("Don't you think so?");
/// assert_eq!(tokens, vec!["DON'T", " ", "YOU", " ", "THINK", " ", "SO", "?"]);
/// ```
pub fn tokenize(input: &str) -> Vec<String> {
    if input.is_empty() {
        return vec![".".to_string()];
    }

    let upper = input.to_uppercase();
    let bytes = upper.as_bytes();
    let mut tokens = Vec::new();
    let mut start = 0;

    for pos in 1..=bytes.len() {
        if is_boundary(bytes, pos) {
            if pos > start {
                let token = String::from_utf8_lossy(&bytes[start..pos]).into_owned();
                tokens.push(token);
            }
            start = pos;
        }
    }

    if tokens.is_empty() {
        return vec![".".to_string()];
    }

    // Sentence-terminal normalization.
    normalize_terminal(&mut tokens);

    tokens
}

/// Determine if position `pos` in `input` is a word boundary.
///
/// Rules (from MEGAHAL_SPEC.md Section 4.1):
/// 1. pos == 0: never a boundary
/// 2. pos == len: always a boundary
/// 3. Apostrophe rule: if char at pos is `'` and both neighbors are alpha, no boundary.
///    If char at pos-1 is `'` and both pos-2 and pos are alpha, no boundary.
/// 4. Alpha transition: exactly one of pos and pos-1 is alphabetic → boundary
/// 5. Digit transition: digit status differs between pos and pos-1 → boundary
fn is_boundary(input: &[u8], pos: usize) -> bool {
    if pos == 0 {
        return false;
    }
    if pos == input.len() {
        return true;
    }

    let curr = input[pos];
    let prev = input[pos - 1];

    // Apostrophe rule: keep contractions together.
    if curr == b'\''
        && pos + 1 < input.len()
        && prev.is_ascii_alphabetic()
        && input[pos + 1].is_ascii_alphabetic()
    {
        return false;
    }
    if prev == b'\''
        && pos >= 2
        && input[pos - 2].is_ascii_alphabetic()
        && curr.is_ascii_alphabetic()
    {
        return false;
    }

    // Alpha transition: exactly one is alphabetic.
    if curr.is_ascii_alphabetic() != prev.is_ascii_alphabetic() {
        return true;
    }

    // Digit transition.
    if curr.is_ascii_digit() != prev.is_ascii_digit() {
        return true;
    }

    false
}

/// Ensure the token sequence ends with sentence-terminal punctuation.
///
/// - If the last token starts with an alphanumeric char, append "."
/// - Otherwise if the last token doesn't end with `!`, `.`, or `?`, replace it with "."
fn normalize_terminal(tokens: &mut Vec<String>) {
    if tokens.is_empty() {
        tokens.push(".".to_string());
        return;
    }

    let last = tokens.last().unwrap();
    let first_byte = last.as_bytes().first().copied();
    let last_byte = last.as_bytes().last().copied();

    if first_byte.is_some_and(|b| b.is_ascii_alphanumeric()) {
        tokens.push(".".to_string());
    } else if last_byte.is_some_and(|b| !matches!(b, b'!' | b'.' | b'?')) {
        *tokens.last_mut().unwrap() = ".".to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_example_dont_you_think_so() {
        let tokens = tokenize("Don't you think so?");
        assert_eq!(
            tokens,
            vec!["DON'T", " ", "YOU", " ", "THINK", " ", "SO", "?"]
        );
    }

    #[test]
    fn simple_sentence() {
        let tokens = tokenize("Hello world");
        assert_eq!(tokens, vec!["HELLO", " ", "WORLD", "."]);
    }

    #[test]
    fn already_terminated_with_period() {
        let tokens = tokenize("Hello.");
        assert_eq!(tokens, vec!["HELLO", "."]);
    }

    #[test]
    fn already_terminated_with_exclamation() {
        let tokens = tokenize("Hello!");
        assert_eq!(tokens, vec!["HELLO", "!"]);
    }

    #[test]
    fn contraction_im() {
        let tokens = tokenize("I'm fine");
        assert_eq!(tokens, vec!["I'M", " ", "FINE", "."]);
    }

    #[test]
    fn digits_split_from_words() {
        let tokens = tokenize("abc123def");
        assert_eq!(tokens, vec!["ABC", "123", "DEF", "."]);
    }

    #[test]
    fn empty_input() {
        let tokens = tokenize("");
        assert_eq!(tokens, vec!["."]);
    }

    #[test]
    fn punctuation_only() {
        let tokens = tokenize("...");
        assert_eq!(tokens, vec!["..."]);
    }

    #[test]
    fn non_terminal_punctuation_replaced() {
        // Comma at the end should be replaced with "."
        let tokens = tokenize("hello,");
        assert_eq!(tokens, vec!["HELLO", "."]);
    }

    #[test]
    fn multiple_spaces_preserved() {
        let tokens = tokenize("A  B");
        assert_eq!(tokens, vec!["A", "  ", "B", "."]);
    }

    #[test]
    fn question_mark_preserved() {
        let tokens = tokenize("Why?");
        assert_eq!(tokens, vec!["WHY", "?"]);
    }

    #[test]
    fn digit_to_punctuation_boundary() {
        // Digit→non-alpha boundary exercises the digit transition path
        // (not caught by alpha transition since neither side is alphabetic).
        let tokens = tokenize("5,");
        assert_eq!(tokens, vec!["5", "."]);
    }

    #[test]
    fn whitespace_only_input() {
        let tokens = tokenize("   ");
        assert_eq!(tokens, vec!["."]);
    }
}
