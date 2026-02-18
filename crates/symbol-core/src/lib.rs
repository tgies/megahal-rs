//! Core Symbol trait and SymbolId types for generic Markov chain models.
//!
//! This crate defines the foundational abstraction for symbols in n-gram tries
//! and interning dictionaries. A symbol can be any type that implements the
//! [`Symbol`] trait — byte strings, integers, enums, or any other ordered type.

use std::fmt::Debug;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

/// A symbol that can be stored in an n-gram trie or interning dictionary.
///
/// This trait is intentionally minimal — no string assumptions, no character
/// classification. The `Ord` bound enables sorted trie children and dictionary
/// indices. The `Hash` bound enables O(1) membership testing in keyword sets.
///
/// The two sentinel methods (`error` and `fin`) are structurally necessary for
/// any Markov model that uses sentence boundaries. They are not application-specific.
pub trait Symbol: Clone + Eq + Ord + Hash + Debug + Send + Sync {
    /// Sentinel value representing "symbol not found" or an error state.
    fn error() -> Self;

    /// Sentinel value representing end-of-sequence (sentence terminator).
    fn fin() -> Self;
}

/// Compact identifier assigned to a symbol by a `SymbolDict`.
///
/// Uses `u16` storage, supporting up to 65,534 unique symbols (IDs 0 and 1
/// are reserved for sentinels). This matches the original MegaHAL limit but
/// is useful for any system needing compact symbol interning.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
pub struct SymbolId(pub u16);

impl SymbolId {
    /// Create a new SymbolId from a raw u16 value.
    #[inline]
    pub fn as_u16(self) -> u16 {
        self.0
    }

    /// Create a SymbolId from a usize index. Panics if index > u16::MAX.
    #[inline]
    pub fn from_usize(index: usize) -> Self {
        assert!(index <= u16::MAX as usize, "SymbolId overflow: {index}");
        SymbolId(index as u16)
    }

    /// Convert to usize for indexing.
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// Sentinel SymbolId for "not found" / error state. Always ID 0.
pub const ERROR_ID: SymbolId = SymbolId(0);

/// Sentinel SymbolId for end-of-sequence. Always ID 1.
pub const FIN_ID: SymbolId = SymbolId(1);

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal Symbol implementation for testing.
    #[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
    struct TestSymbol(u32);

    impl Symbol for TestSymbol {
        fn error() -> Self {
            TestSymbol(u32::MAX)
        }
        fn fin() -> Self {
            TestSymbol(u32::MAX - 1)
        }
    }

    #[test]
    fn sentinel_values_are_distinct() {
        assert_ne!(TestSymbol::error(), TestSymbol::fin());
    }

    #[test]
    fn sentinel_ids_are_correct() {
        assert_eq!(ERROR_ID, SymbolId(0));
        assert_eq!(FIN_ID, SymbolId(1));
        assert_ne!(ERROR_ID, FIN_ID);
    }

    #[test]
    fn symbol_id_roundtrip() {
        let id = SymbolId(42);
        assert_eq!(id.as_u16(), 42);
        assert_eq!(id.as_usize(), 42);
    }

    #[test]
    fn symbol_id_from_usize() {
        let id = SymbolId::from_usize(100);
        assert_eq!(id, SymbolId(100));
    }

    #[test]
    #[should_panic(expected = "SymbolId overflow")]
    fn symbol_id_overflow_panics() {
        SymbolId::from_usize(u16::MAX as usize + 1);
    }

    #[test]
    fn symbol_id_ordering() {
        assert!(SymbolId(0) < SymbolId(1));
        assert!(SymbolId(1) < SymbolId(65535));
    }

    #[test]
    fn symbol_id_serde_roundtrip() {
        let id = SymbolId(42);
        let json = serde_json::to_string(&id).unwrap();
        let back: SymbolId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }
}
