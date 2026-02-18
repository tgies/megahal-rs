//! Generic interning dictionary mapping [`Symbol`] values to compact [`SymbolId`] identifiers.
//!
//! The dictionary maintains two parallel structures:
//! - `entries`: symbols in insertion order (index = SymbolId)
//! - `sorted_index`: SymbolIds sorted alphabetically by their symbol, for O(log n) lookup
//!
//! This design matches the original MegaHAL dictionary exactly, but is generic over
//! any [`Symbol`] type. It is independently useful as a string interner or any
//! system needing compact, ordered symbol interning.
//!
//! Sentinel entries `ERROR_ID` (0) and `FIN_ID` (1) are pre-populated at construction.

use serde::{Deserialize, Serialize};
use symbol_core::{ERROR_ID, FIN_ID, Symbol, SymbolId};

/// An interning dictionary that maps symbols to compact [`SymbolId`] values.
///
/// Symbols are assigned sequential IDs starting from 0. IDs 0 and 1 are always
/// reserved for the [`ERROR_ID`] and [`FIN_ID`] sentinels respectively.
///
/// Lookup is O(log n) via binary search over a sorted index.
/// Insertion is O(n) in the worst case (due to index shifting), but amortized
/// O(log n) for the search component.
#[derive(Debug, Serialize, Deserialize)]
pub struct SymbolDict<S: Symbol> {
    /// Symbols in insertion order. `entries[id.as_usize()]` returns the symbol for `id`.
    entries: Vec<S>,
    /// Indices into `entries`, kept sorted by the symbol they reference.
    /// Used for O(log n) binary search during lookup/insertion.
    sorted_index: Vec<SymbolId>,
}

impl<S: Symbol> SymbolDict<S> {
    /// Create a new dictionary pre-populated with the two sentinel entries:
    /// - ID 0: `S::error()`
    /// - ID 1: `S::fin()`
    pub fn new() -> Self {
        let error = S::error();
        let fin = S::fin();

        let mut dict = SymbolDict {
            entries: vec![error.clone(), fin.clone()],
            sorted_index: Vec::with_capacity(2),
        };

        // Build sorted index for the two sentinels.
        if error <= fin {
            dict.sorted_index.push(ERROR_ID);
            dict.sorted_index.push(FIN_ID);
        } else {
            dict.sorted_index.push(FIN_ID);
            dict.sorted_index.push(ERROR_ID);
        }

        dict
    }

    /// Insert a symbol into the dictionary if not already present, returning its ID.
    ///
    /// If the symbol already exists, returns the existing ID without duplicating.
    /// New symbols are assigned the next sequential ID.
    pub fn intern(&mut self, symbol: S) -> SymbolId {
        // Binary search the sorted index for this symbol.
        let search_result = self
            .sorted_index
            .binary_search_by(|&id| self.entries[id.as_usize()].cmp(&symbol));

        match search_result {
            Ok(idx) => {
                // Already exists — return existing ID.
                self.sorted_index[idx]
            }
            Err(insert_pos) => {
                // New symbol — assign next ID.
                let new_id = SymbolId::from_usize(self.entries.len());
                self.entries.push(symbol);
                self.sorted_index.insert(insert_pos, new_id);
                new_id
            }
        }
    }

    /// Look up a symbol without inserting. Returns `None` if absent.
    pub fn find(&self, symbol: &S) -> Option<SymbolId> {
        self.sorted_index
            .binary_search_by(|&id| self.entries[id.as_usize()].cmp(symbol))
            .ok()
            .map(|idx| self.sorted_index[idx])
    }

    /// Resolve a SymbolId back to the symbol it represents.
    ///
    /// # Panics
    /// Panics if `id` is out of bounds.
    #[inline]
    pub fn resolve(&self, id: SymbolId) -> &S {
        &self.entries[id.as_usize()]
    }

    /// Number of symbols in the dictionary (including sentinels).
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the dictionary contains only sentinel entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.len() <= 2
    }
}

impl<S: Symbol> Default for SymbolDict<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde::{Deserialize, Serialize};

    /// A simple Symbol for testing: case-insensitive ASCII string.
    #[derive(Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
    struct TestSym(String);

    impl PartialOrd for TestSym {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for TestSym {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.to_uppercase().cmp(&other.0.to_uppercase())
        }
    }

    impl Symbol for TestSym {
        fn error() -> Self {
            TestSym("<ERROR>".to_string())
        }
        fn fin() -> Self {
            TestSym("<FIN>".to_string())
        }
    }

    #[test]
    fn new_dict_has_sentinels() {
        let dict = SymbolDict::<TestSym>::new();
        assert_eq!(dict.len(), 2);
        assert_eq!(dict.resolve(ERROR_ID), &TestSym::error());
        assert_eq!(dict.resolve(FIN_ID), &TestSym::fin());
    }

    #[test]
    fn intern_returns_sequential_ids() {
        let mut dict = SymbolDict::<TestSym>::new();
        let id_hello = dict.intern(TestSym("HELLO".into()));
        let id_world = dict.intern(TestSym("WORLD".into()));

        assert_eq!(id_hello, SymbolId(2)); // 0 and 1 are sentinels
        assert_eq!(id_world, SymbolId(3));
        assert_eq!(dict.len(), 4);
    }

    #[test]
    fn intern_deduplicates() {
        let mut dict = SymbolDict::<TestSym>::new();
        let first = dict.intern(TestSym("HELLO".into()));
        let second = dict.intern(TestSym("HELLO".into()));

        assert_eq!(first, second);
        assert_eq!(dict.len(), 3); // 2 sentinels + 1 word
    }

    #[test]
    fn find_existing() {
        let mut dict = SymbolDict::<TestSym>::new();
        let id = dict.intern(TestSym("TEST".into()));
        assert_eq!(dict.find(&TestSym("TEST".into())), Some(id));
    }

    #[test]
    fn find_missing() {
        let dict = SymbolDict::<TestSym>::new();
        assert_eq!(dict.find(&TestSym("NOPE".into())), None);
    }

    #[test]
    fn resolve_roundtrip() {
        let mut dict = SymbolDict::<TestSym>::new();
        let sym = TestSym("ROUNDTRIP".into());
        let id = dict.intern(sym.clone());
        assert_eq!(dict.resolve(id), &sym);
    }

    #[test]
    fn case_insensitive_lookup() {
        let mut dict = SymbolDict::<TestSym>::new();
        let id = dict.intern(TestSym("hello".into()));

        // Our TestSym compares case-insensitively.
        assert_eq!(dict.find(&TestSym("HELLO".into())), Some(id));
    }

    #[test]
    fn dict_serde_roundtrip() {
        let mut dict = SymbolDict::<TestSym>::new();
        dict.intern(TestSym("HELLO".into()));
        dict.intern(TestSym("WORLD".into()));
        dict.intern(TestSym("APPLE".into()));

        let json = serde_json::to_string(&dict).unwrap();
        let back: SymbolDict<TestSym> = serde_json::from_str(&json).unwrap();

        assert_eq!(back.len(), dict.len());
        assert!(back.find(&TestSym("HELLO".into())).is_some());
        assert!(back.find(&TestSym("WORLD".into())).is_some());
        assert!(back.find(&TestSym("APPLE".into())).is_some());
    }

    /// A Symbol where error() > fin() to exercise the else branch in new().
    #[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
    struct ReversedSentinelSym(u32);

    impl Symbol for ReversedSentinelSym {
        fn error() -> Self {
            ReversedSentinelSym(u32::MAX) // error > fin
        }
        fn fin() -> Self {
            ReversedSentinelSym(0)
        }
    }

    #[test]
    fn new_dict_error_greater_than_fin() {
        let dict = SymbolDict::<ReversedSentinelSym>::new();
        assert_eq!(dict.len(), 2);
        // Both sentinels should be findable despite reversed ordering.
        assert_eq!(dict.resolve(ERROR_ID), &ReversedSentinelSym::error());
        assert_eq!(dict.resolve(FIN_ID), &ReversedSentinelSym::fin());
        assert!(dict.find(&ReversedSentinelSym::error()).is_some());
        assert!(dict.find(&ReversedSentinelSym::fin()).is_some());
    }

    #[test]
    fn is_empty_on_new_dict() {
        let dict = SymbolDict::<TestSym>::new();
        assert!(dict.is_empty());
    }

    #[test]
    fn is_empty_after_intern() {
        let mut dict = SymbolDict::<TestSym>::new();
        dict.intern(TestSym("WORD".into()));
        assert!(!dict.is_empty());
    }

    #[test]
    fn default_creates_empty_dict() {
        let dict = SymbolDict::<TestSym>::default();
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 2);
    }

    #[test]
    fn sorted_index_maintained() {
        let mut dict = SymbolDict::<TestSym>::new();
        // Insert in non-alphabetical order.
        dict.intern(TestSym("ZEBRA".into()));
        dict.intern(TestSym("APPLE".into()));
        dict.intern(TestSym("MANGO".into()));

        // All should be findable (proves sorted index is correct).
        assert!(dict.find(&TestSym("ZEBRA".into())).is_some());
        assert!(dict.find(&TestSym("APPLE".into())).is_some());
        assert!(dict.find(&TestSym("MANGO".into())).is_some());
    }
}
