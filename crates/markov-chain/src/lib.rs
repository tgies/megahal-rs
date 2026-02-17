//! Bidirectional Markov model with context window for n-gram learning and generation.
//!
//! This crate provides [`BidirectionalModel`], which combines a forward trie,
//! backward trie, and symbol dictionary into a complete bidirectional Markov model.
//! The [`ContextWindow`] type manages the sliding context state used during both
//! learning (mutating the trie) and generation/evaluation (read-only traversal).
//!
//! The model is generic over any [`Symbol`] type, making it reusable for any
//! n-gram modeling task — not just chatbots.

use ngram_trie::{NodeRef, Trie};
use serde::{Deserialize, Serialize};
use symbol_core::{Symbol, SymbolId, FIN_ID};
use symbol_dict::SymbolDict;

/// A sliding context window tracking position in an n-gram trie.
///
/// Stores `order + 2` slots (indices 0 through `order + 1`), matching the
/// original MegaHAL context array. Slot 0 is always the trie root. Deeper
/// slots track progressively longer context paths.
///
/// Because slots store [`NodeRef`] indices (not borrows), the window is safe
/// to use alongside `&mut Trie` — no aliasing conflicts.
#[derive(Debug, Clone)]
pub struct ContextWindow {
    slots: Vec<Option<NodeRef>>,
    order: u8,
}

impl ContextWindow {
    /// Create a new context window for a model of the given order.
    /// All slots are initialized to `None`.
    pub fn new(order: u8) -> Self {
        ContextWindow {
            slots: vec![None; order as usize + 2],
            order,
        }
    }

    /// Reset the window: set slot[0] to `root`, all others to `None`.
    pub fn initialize(&mut self, root: NodeRef) {
        for slot in &mut self.slots {
            *slot = None;
        }
        self.slots[0] = Some(root);
    }

    /// Read-only advance: update context by finding (not creating) children.
    ///
    /// Used during generation and evaluation. Walks from depth `order + 1`
    /// down to 1: if `slots[d-1]` is non-None, look up the symbol as a child
    /// and store the result (which may be `None`) in `slots[d]`.
    pub fn advance(&mut self, trie: &Trie, symbol: SymbolId) {
        for d in (1..=self.order as usize + 1).rev() {
            if let Some(parent) = self.slots[d - 1] {
                self.slots[d] = trie.find_child(parent, symbol);
            }
        }
    }

    /// Mutating advance: update context by finding or creating children.
    ///
    /// Used during learning. Same traversal as [`advance`], but uses
    /// `Trie::add_child` which creates new nodes and increments counts.
    pub fn advance_and_learn(&mut self, trie: &mut Trie, symbol: SymbolId) {
        for d in (1..=self.order as usize + 1).rev() {
            if let Some(parent) = self.slots[d - 1] {
                self.slots[d] = Some(trie.add_child(parent, symbol));
            }
        }
    }

    /// Get the context node at depth `j` (0-indexed).
    ///
    /// For evaluation, `j` ranges from 0 to `order - 1` (the deepest level
    /// used during training is excluded from scoring).
    #[inline]
    pub fn at_depth(&self, j: usize) -> Option<NodeRef> {
        self.slots.get(j).copied().flatten()
    }

    /// Get the deepest non-None context node.
    ///
    /// Used by the babble function to select the most specific context.
    /// Scans from slot 0 up to slot `order`, returning the last non-None.
    pub fn deepest(&self) -> Option<NodeRef> {
        let mut best = None;
        for d in 0..=self.order as usize {
            if self.slots[d].is_some() {
                best = self.slots[d];
            }
        }
        best
    }

    /// The model order this window was created for.
    #[inline]
    pub fn order(&self) -> u8 {
        self.order
    }
}

/// A bidirectional Markov model: forward trie + backward trie + shared dictionary.
///
/// Learning trains both tries from the same token sequence — forward (left-to-right)
/// and backward (right-to-left). This enables bidirectional reply generation where
/// a seed word can be expanded both forward to the end of a sentence and backward
/// to the beginning.
#[derive(Debug, Serialize, Deserialize)]
pub struct BidirectionalModel<S: Symbol> {
    /// Model order (trie depth). Default: 5.
    pub order: u8,
    /// Forward trie: models left-to-right token sequences.
    pub forward: Trie,
    /// Backward trie: models right-to-left token sequences.
    pub backward: Trie,
    /// Shared dictionary mapping symbols to compact IDs.
    pub dictionary: SymbolDict<S>,
}

impl<S: Symbol> BidirectionalModel<S> {
    /// Create a new empty model with the given order.
    pub fn new(order: u8) -> Self {
        BidirectionalModel {
            order,
            forward: Trie::new(),
            backward: Trie::new(),
            dictionary: SymbolDict::new(),
        }
    }

    /// Learn from a token sequence, updating both forward and backward tries.
    ///
    /// Skips entirely if `tokens.len() <= order` (too short for meaningful context).
    ///
    /// Forward pass: iterates tokens left-to-right, adding each to the dictionary
    /// and updating the forward trie. Finishes with `FIN_ID`.
    ///
    /// Backward pass: iterates tokens right-to-left, looking up (not adding) each
    /// in the dictionary. Finishes with `FIN_ID`.
    pub fn learn(&mut self, tokens: &[S]) {
        if tokens.len() <= self.order as usize {
            return;
        }

        // Forward pass: add symbols to dictionary and train forward trie.
        let mut ctx = ContextWindow::new(self.order);
        ctx.initialize(self.forward.root());

        let symbol_ids: Vec<SymbolId> = tokens
            .iter()
            .map(|tok| self.dictionary.intern(tok.clone()))
            .collect();

        for &id in &symbol_ids {
            ctx.advance_and_learn(&mut self.forward, id);
        }
        ctx.advance_and_learn(&mut self.forward, FIN_ID);

        // Backward pass: use existing dictionary IDs, train backward trie.
        let mut ctx = ContextWindow::new(self.order);
        ctx.initialize(self.backward.root());

        for &id in symbol_ids.iter().rev() {
            ctx.advance_and_learn(&mut self.backward, id);
        }
        ctx.advance_and_learn(&mut self.backward, FIN_ID);
    }

    /// Create a context window initialized to the forward trie root.
    pub fn forward_context(&self) -> ContextWindow {
        let mut ctx = ContextWindow::new(self.order);
        ctx.initialize(self.forward.root());
        ctx
    }

    /// Create a context window initialized to the backward trie root.
    pub fn backward_context(&self) -> ContextWindow {
        let mut ctx = ContextWindow::new(self.order);
        ctx.initialize(self.backward.root());
        ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
    struct TestSym(String);

    impl Symbol for TestSym {
        fn error() -> Self {
            TestSym("<ERROR>".into())
        }
        fn fin() -> Self {
            TestSym("<FIN>".into())
        }
    }

    fn make_tokens(words: &[&str]) -> Vec<TestSym> {
        words.iter().map(|w| TestSym(w.to_string())).collect()
    }

    #[test]
    fn context_window_initialize() {
        let trie = Trie::new();
        let mut ctx = ContextWindow::new(5);
        ctx.initialize(trie.root());

        assert_eq!(ctx.at_depth(0), Some(trie.root()));
        assert_eq!(ctx.at_depth(1), None);
        assert_eq!(ctx.at_depth(6), None); // order + 1
    }

    #[test]
    fn context_window_deepest() {
        let mut trie = Trie::new();
        let mut ctx = ContextWindow::new(5);
        ctx.initialize(trie.root());

        // Initially, only depth 0 has a node.
        assert_eq!(ctx.deepest(), Some(trie.root()));

        // After advancing, deeper slots get filled.
        let child = trie.add_child(trie.root(), SymbolId(2));
        ctx.advance(&trie, SymbolId(2));
        assert_eq!(ctx.at_depth(1), Some(child));

        // Deepest should be the child now.
        let deepest = ctx.deepest().unwrap();
        assert_eq!(deepest, child);
    }

    #[test]
    fn learn_skips_short_input() {
        let mut model = BidirectionalModel::<TestSym>::new(5);
        let short = make_tokens(&["A", " ", "B"]);

        model.learn(&short);

        // Nothing should have been added beyond sentinels.
        assert_eq!(model.dictionary.len(), 2); // only ERROR and FIN
        assert!(model.forward.is_empty());
    }

    #[test]
    fn learn_populates_dictionary() {
        let mut model = BidirectionalModel::<TestSym>::new(2);
        let tokens = make_tokens(&["THE", " ", "CAT"]);

        model.learn(&tokens);

        // 2 sentinels + 3 tokens = 5
        assert_eq!(model.dictionary.len(), 5);
        assert!(model.dictionary.find(&TestSym("THE".into())).is_some());
        assert!(model.dictionary.find(&TestSym("CAT".into())).is_some());
        assert!(model.dictionary.find(&TestSym(" ".into())).is_some());
    }

    #[test]
    fn learn_populates_forward_trie() {
        let mut model = BidirectionalModel::<TestSym>::new(2);
        let tokens = make_tokens(&["A", "B", "C"]);
        model.learn(&tokens);

        // Forward trie should have root → A, root → A → B, etc.
        let root = model.forward.root();
        let id_a = model.dictionary.find(&TestSym("A".into())).unwrap();
        let child_a = model.forward.find_child(root, id_a);
        assert!(child_a.is_some());
    }

    #[test]
    fn learn_populates_backward_trie() {
        let mut model = BidirectionalModel::<TestSym>::new(2);
        let tokens = make_tokens(&["A", "B", "C"]);
        model.learn(&tokens);

        // Backward trie should have root → C (since backward iterates right-to-left).
        let root = model.backward.root();
        let id_c = model.dictionary.find(&TestSym("C".into())).unwrap();
        let child_c = model.backward.find_child(root, id_c);
        assert!(child_c.is_some());
    }

    #[test]
    fn forward_and_backward_context_creation() {
        let model = BidirectionalModel::<TestSym>::new(5);
        let fwd = model.forward_context();
        let bwd = model.backward_context();

        assert_eq!(fwd.at_depth(0), Some(model.forward.root()));
        assert_eq!(bwd.at_depth(0), Some(model.backward.root()));
        assert_eq!(fwd.order(), 5);
        assert_eq!(bwd.order(), 5);
    }

    #[test]
    fn model_serde_roundtrip() {
        let mut model = BidirectionalModel::<TestSym>::new(2);
        model.learn(&make_tokens(&["A", "B", "C"]));
        model.learn(&make_tokens(&["X", "Y", "Z"]));

        let json = serde_json::to_string(&model).unwrap();
        let back: BidirectionalModel<TestSym> = serde_json::from_str(&json).unwrap();

        assert_eq!(back.order, 2);
        assert_eq!(back.dictionary.len(), model.dictionary.len());
        assert!(back.dictionary.find(&TestSym("A".into())).is_some());
        assert!(back.dictionary.find(&TestSym("Z".into())).is_some());

        let root = back.forward.root();
        let id_a = back.dictionary.find(&TestSym("A".into())).unwrap();
        assert!(back.forward.find_child(root, id_a).is_some());

        let broot = back.backward.root();
        let id_c = back.dictionary.find(&TestSym("C".into())).unwrap();
        assert!(back.backward.find_child(broot, id_c).is_some());
    }

    #[test]
    fn frequency_access_pattern() {
        // Verify the evaluation-style access: walk context, read count/usage at each depth.
        let mut model = BidirectionalModel::<TestSym>::new(2);
        // Learn "A B" multiple times to build up counts.
        for _ in 0..10 {
            model.learn(&make_tokens(&["A", "B", "C"]));
        }

        let mut ctx = model.forward_context();
        let id_a = model.dictionary.find(&TestSym("A".into())).unwrap();
        let id_b = model.dictionary.find(&TestSym("B".into())).unwrap();

        ctx.advance(&model.forward, id_a);

        // At depth 0 (root), B should be findable with count > 0.
        let root = ctx.at_depth(0).unwrap();
        if let Some(child_b) = model.forward.find_child(root, id_b) {
            let child = model.forward.node(child_b);
            let parent = model.forward.node(root);
            assert!(child.count > 0);
            assert!(parent.usage > 0);
            let _prob = child.count as f64 / parent.usage as f64;
        }

        // At depth 1 (after seeing A), B should also be findable.
        if let Some(parent_at_1) = ctx.at_depth(1) {
            if let Some(child_b) = model.forward.find_child(parent_at_1, id_b) {
                let child = model.forward.node(child_b);
                let parent = model.forward.node(parent_at_1);
                let prob = child.count as f64 / parent.usage as f64;
                // After "A", "B" should be highly probable.
                assert!(prob > 0.0);
            }
        }
    }
}
