//! Arena-based n-gram frequency trie with sorted children and saturating counts.
//!
//! This crate provides a generic trie data structure for storing n-gram frequency
//! statistics. Nodes are stored in a contiguous arena (`Vec<TrieNode>`) and
//! referenced by opaque [`NodeRef`] handles, which are plain indices. This design
//! avoids borrow checker issues when maintaining a context window alongside a
//! mutable trie — indices don't borrow the arena.
//!
//! Children of each node are kept sorted by [`SymbolId`] for O(log n) binary
//! search. Counts saturate at `u16::MAX` (65535) — once reached, neither the
//! node's count nor its parent's usage are incremented further.

use serde::{Deserialize, Serialize};
use symbol_core::SymbolId;

/// Opaque handle into the trie's node arena.
///
/// This is a plain index — it does not borrow the trie. You can hold arbitrarily
/// many `NodeRef` values while mutating the trie, which is essential for the
/// context window pattern used during learning.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct NodeRef(u32);

impl NodeRef {
    /// Convert to usize for indexing into the arena.
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Create from a usize index.
    #[inline]
    fn from_usize(index: usize) -> Self {
        NodeRef(index as u32)
    }
}

/// A single node in the n-gram frequency trie.
///
/// Each node represents a symbol observed in a particular context. The `count`
/// field records how many times this symbol appeared as the "next symbol" in its
/// parent's context. The `usage` field records the total count of all observations
/// through this node (sum of all children's counts), used as the denominator when
/// computing transition probabilities.
///
/// Fields are public to allow direct access for probability computation:
/// `P(child|parent) = child.count / parent.usage`
#[derive(Debug, Serialize, Deserialize)]
pub struct TrieNode {
    /// The symbol ID this node represents.
    pub symbol: SymbolId,
    /// Total observations through this node (sum of children's counts).
    pub usage: u32,
    /// How many times this symbol was observed in its parent's context.
    /// Saturates at u16::MAX.
    pub count: u16,
    /// Child node references, kept sorted by symbol ID.
    children: Vec<NodeRef>,
}

impl TrieNode {
    fn new(symbol: SymbolId) -> Self {
        TrieNode {
            symbol,
            usage: 0,
            count: 0,
            children: Vec::new(),
        }
    }
}

/// Arena-based n-gram frequency trie.
///
/// All nodes are stored in a contiguous `Vec`, referenced by [`NodeRef`] indices.
/// The root node is always at index 0.
///
/// # Example
///
/// ```
/// use ngram_trie::Trie;
/// use symbol_core::SymbolId;
///
/// let mut trie = Trie::new();
/// let root = trie.root();
///
/// // Learning: add a child symbol, incrementing counts.
/// let child = trie.add_child(root, SymbolId(2));
/// assert_eq!(trie.node(child).count, 1);
/// assert_eq!(trie.node(root).usage, 1);
///
/// // Repeat to increment.
/// let same_child = trie.add_child(root, SymbolId(2));
/// assert_eq!(same_child, child);
/// assert_eq!(trie.node(child).count, 2);
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct Trie {
    nodes: Vec<TrieNode>,
}

impl Trie {
    /// Create a new trie with a single root node.
    /// The root has symbol ERROR_ID (0) and represents the empty context.
    pub fn new() -> Self {
        let root = TrieNode::new(SymbolId(0));
        Trie { nodes: vec![root] }
    }

    /// Get a reference to the root node.
    #[inline]
    pub fn root(&self) -> NodeRef {
        NodeRef(0)
    }

    /// Access a node by reference.
    #[inline]
    pub fn node(&self, r: NodeRef) -> &TrieNode {
        &self.nodes[r.as_usize()]
    }

    /// Find an existing child of `parent` matching `symbol`.
    /// Returns `None` if no such child exists.
    pub fn find_child(&self, parent: NodeRef, symbol: SymbolId) -> Option<NodeRef> {
        let parent_node = &self.nodes[parent.as_usize()];
        let children = &parent_node.children;

        // Binary search over children sorted by symbol ID.
        children
            .binary_search_by(|child_ref| self.nodes[child_ref.as_usize()].symbol.cmp(&symbol))
            .ok()
            .map(|idx| children[idx])
    }

    /// Find or create a child of `parent` matching `symbol`.
    ///
    /// If the child already exists, its `count` is incremented (saturating at
    /// `u16::MAX`). If `count` is already at `u16::MAX`, neither `count` nor
    /// the parent's `usage` are incremented. If the child is new, it is created
    /// with `count = 1`.
    ///
    /// The parent's `usage` is incremented alongside the child's `count`
    /// (also subject to saturation).
    ///
    /// Returns a reference to the child node.
    pub fn add_child(&mut self, parent: NodeRef, symbol: SymbolId) -> NodeRef {
        let parent_node = &self.nodes[parent.as_usize()];

        // Binary search for insertion point.
        let search_result = parent_node
            .children
            .binary_search_by(|child_ref| self.nodes[child_ref.as_usize()].symbol.cmp(&symbol));

        match search_result {
            Ok(idx) => {
                // Child exists — increment counts with saturation.
                let child_ref = self.nodes[parent.as_usize()].children[idx];
                let child = &mut self.nodes[child_ref.as_usize()];
                if child.count < u16::MAX {
                    child.count += 1;
                    self.nodes[parent.as_usize()].usage += 1;
                }
                child_ref
            }
            Err(idx) => {
                // Child doesn't exist — create it.
                let child_ref = NodeRef::from_usize(self.nodes.len());
                self.nodes.push(TrieNode::new(symbol));
                self.nodes[child_ref.as_usize()].count = 1;
                self.nodes[parent.as_usize()].usage += 1;
                self.nodes[parent.as_usize()].children.insert(idx, child_ref);
                child_ref
            }
        }
    }

    /// Get the child references of a node (sorted by symbol ID).
    #[inline]
    pub fn children(&self, parent: NodeRef) -> &[NodeRef] {
        &self.nodes[parent.as_usize()].children
    }

    /// Number of children of a node.
    #[inline]
    pub fn branch_count(&self, parent: NodeRef) -> usize {
        self.nodes[parent.as_usize()].children.len()
    }

    /// Total number of nodes in the trie (including root).
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the trie contains only the root node.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.len() == 1
    }
}

impl Default for Trie {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_trie_has_root() {
        let trie = Trie::new();
        let root = trie.root();
        let node = trie.node(root);
        assert_eq!(node.symbol, SymbolId(0));
        assert_eq!(node.usage, 0);
        assert_eq!(node.count, 0);
        assert!(trie.children(root).is_empty());
    }

    #[test]
    fn add_child_creates_new_node() {
        let mut trie = Trie::new();
        let root = trie.root();
        let child = trie.add_child(root, SymbolId(5));

        assert_eq!(trie.node(child).symbol, SymbolId(5));
        assert_eq!(trie.node(child).count, 1);
        assert_eq!(trie.node(root).usage, 1);
        assert_eq!(trie.branch_count(root), 1);
    }

    #[test]
    fn add_child_increments_existing() {
        let mut trie = Trie::new();
        let root = trie.root();

        let first = trie.add_child(root, SymbolId(5));
        let second = trie.add_child(root, SymbolId(5));

        assert_eq!(first, second);
        assert_eq!(trie.node(first).count, 2);
        assert_eq!(trie.node(root).usage, 2);
        assert_eq!(trie.branch_count(root), 1); // still one child
    }

    #[test]
    fn children_are_sorted_by_symbol() {
        let mut trie = Trie::new();
        let root = trie.root();

        // Add children in non-sorted order.
        trie.add_child(root, SymbolId(10));
        trie.add_child(root, SymbolId(3));
        trie.add_child(root, SymbolId(7));
        trie.add_child(root, SymbolId(1));

        let children = trie.children(root);
        let symbols: Vec<SymbolId> = children
            .iter()
            .map(|&r| trie.node(r).symbol)
            .collect();

        assert_eq!(symbols, vec![SymbolId(1), SymbolId(3), SymbolId(7), SymbolId(10)]);
    }

    #[test]
    fn find_child_existing() {
        let mut trie = Trie::new();
        let root = trie.root();
        let added = trie.add_child(root, SymbolId(42));

        let found = trie.find_child(root, SymbolId(42));
        assert_eq!(found, Some(added));
    }

    #[test]
    fn find_child_missing() {
        let trie = Trie::new();
        let root = trie.root();
        assert_eq!(trie.find_child(root, SymbolId(99)), None);
    }

    #[test]
    fn count_saturation_at_u16_max() {
        let mut trie = Trie::new();
        let root = trie.root();

        // Manually set count close to max.
        let child = trie.add_child(root, SymbolId(1)); // count = 1, usage = 1
        trie.nodes[child.as_usize()].count = u16::MAX - 1;
        trie.nodes[root.as_usize()].usage = u16::MAX as u32 - 1;

        // One more increment should work.
        trie.add_child(root, SymbolId(1));
        assert_eq!(trie.node(child).count, u16::MAX);
        assert_eq!(trie.node(root).usage, u16::MAX as u32);

        // Further increments should be silently dropped.
        trie.add_child(root, SymbolId(1));
        assert_eq!(trie.node(child).count, u16::MAX);
        assert_eq!(trie.node(root).usage, u16::MAX as u32);
    }

    #[test]
    fn multi_level_trie() {
        let mut trie = Trie::new();
        let root = trie.root();

        let level1 = trie.add_child(root, SymbolId(2));
        let level2 = trie.add_child(level1, SymbolId(3));
        let level3 = trie.add_child(level2, SymbolId(4));

        assert_eq!(trie.node(level3).symbol, SymbolId(4));
        assert_eq!(trie.node(level3).count, 1);
        assert_eq!(trie.node(level2).usage, 1);

        // Navigate back down.
        let found = trie.find_child(root, SymbolId(2)).unwrap();
        let found = trie.find_child(found, SymbolId(3)).unwrap();
        let found = trie.find_child(found, SymbolId(4)).unwrap();
        assert_eq!(found, level3);
    }

    #[test]
    fn trie_serde_roundtrip() {
        let mut trie = Trie::new();
        let root = trie.root();
        trie.add_child(root, SymbolId(2));
        trie.add_child(root, SymbolId(5));
        trie.add_child(root, SymbolId(2)); // increment count

        let json = serde_json::to_string(&trie).unwrap();
        let back: Trie = serde_json::from_str(&json).unwrap();

        let back_root = back.root();
        assert_eq!(back.branch_count(back_root), 2);
        assert_eq!(back.node(back_root).usage, 3);

        let child2 = back.find_child(back_root, SymbolId(2)).unwrap();
        assert_eq!(back.node(child2).count, 2);

        let child5 = back.find_child(back_root, SymbolId(5)).unwrap();
        assert_eq!(back.node(child5).count, 1);
    }

    #[test]
    fn usage_tracks_multiple_children() {
        let mut trie = Trie::new();
        let root = trie.root();

        trie.add_child(root, SymbolId(1));
        trie.add_child(root, SymbolId(2));
        trie.add_child(root, SymbolId(3));
        trie.add_child(root, SymbolId(1)); // increment existing

        assert_eq!(trie.node(root).usage, 4);
        assert_eq!(trie.branch_count(root), 3);
    }
}
