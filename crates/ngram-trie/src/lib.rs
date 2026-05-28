//! Arena-based n-gram frequency trie with sorted children and saturating counts.
//!
//! statistics. Nodes are stored in a contiguous arena (`Vec<TrieNode>`) and
//! referenced by [`NodeRef`] handles.
//!
//! Children of each node are kept sorted by [`SymbolId`] for O(log n) binary
//! search. Counts saturate at `u16::MAX` (65535); once reached, neither the
//! node's count nor its parent's usage are incremented further.

use serde::{Deserialize, Serialize};
use symbol_core::{ERROR_ID, SymbolId};

/// Opaque handle into the trie's node arena.
///
/// This is an index into the trie's arena.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct NodeRef(u32);

impl NodeRef {
    /// Convert to usize for indexing into the arena.
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Create from a usize index.
    ///
    /// Public for use by format importers that need to assemble a trie's child
    /// references before the underlying node arena is fully populated. Most
    /// callers should construct trees via [`Trie::add_child`] instead.
    #[inline]
    pub fn from_usize(index: usize) -> Self {
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

    /// Construct a node from raw field values.
    ///
    /// Intended for format importers (e.g. loading C MegaHAL `.brn` files)
    /// that need to materialize a node with exact field values bypassing the
    /// usage/count accounting performed by [`Trie::add_child`]. `children`
    /// must be sorted by the [`SymbolId`] of the referenced child nodes; the
    /// invariant is checked by [`Trie::from_raw_nodes`] in debug builds.
    pub fn from_raw(symbol: SymbolId, usage: u32, count: u16, children: Vec<NodeRef>) -> Self {
        TrieNode {
            symbol,
            usage,
            count,
            children,
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
/// let child = trie.add_child(root, SymbolId::new(2));
/// assert_eq!(trie.node(child).count, 1);
/// assert_eq!(trie.node(root).usage, 1);
///
/// // Repeat to increment.
/// let same_child = trie.add_child(root, SymbolId::new(2));
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
        let root = TrieNode::new(ERROR_ID);
        Trie { nodes: vec![root] }
    }

    /// Construct a trie from a pre-built node arena.
    ///
    /// Intended for format importers that build the node arena directly (e.g.
    /// loading C MegaHAL `.brn` files). The first element of `nodes` is treated
    /// as the root. In debug builds, this checks that:
    ///
    /// * `nodes` is non-empty,
    /// * every [`NodeRef`] in any node's `children` is in-bounds, and
    /// * each node's children are sorted by the [`SymbolId`] of the referenced
    ///   child node (the invariant relied upon by [`Self::find_child`]).
    ///
    /// Callers must uphold these invariants in release builds.
    pub fn from_raw_nodes(nodes: Vec<TrieNode>) -> Self {
        debug_assert!(!nodes.is_empty(), "from_raw_nodes requires at least a root");
        if cfg!(debug_assertions) {
            for (idx, node) in nodes.iter().enumerate() {
                let mut prev: Option<SymbolId> = None;
                for child_ref in &node.children {
                    let child_idx = child_ref.as_usize();
                    debug_assert!(
                        child_idx < nodes.len(),
                        "node {idx} references out-of-bounds child {child_idx}"
                    );
                    let child_symbol = nodes[child_idx].symbol;
                    if let Some(prev_symbol) = prev {
                        debug_assert!(
                            prev_symbol < child_symbol,
                            "node {idx} children are not strictly sorted by SymbolId"
                        );
                    }
                    prev = Some(child_symbol);
                }
            }
        }
        Trie { nodes }
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
                // Child exists; increment counts with saturation.
                let child_ref = self.nodes[parent.as_usize()].children[idx];
                let child = &mut self.nodes[child_ref.as_usize()];
                if child.count < u16::MAX {
                    child.count += 1;
                    self.nodes[parent.as_usize()].usage += 1;
                }
                child_ref
            }
            Err(idx) => {
                // Child doesn't exist; create it.
                let child_ref = NodeRef::from_usize(self.nodes.len());
                self.nodes.push(TrieNode::new(symbol));
                self.nodes[child_ref.as_usize()].count = 1;
                self.nodes[parent.as_usize()].usage += 1;
                self.nodes[parent.as_usize()]
                    .children
                    .insert(idx, child_ref);
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
        assert_eq!(node.symbol, SymbolId::new(0));
        assert_eq!(node.usage, 0);
        assert_eq!(node.count, 0);
        assert!(trie.children(root).is_empty());
    }

    #[test]
    fn add_child_creates_new_node() {
        let mut trie = Trie::new();
        let root = trie.root();
        let child = trie.add_child(root, SymbolId::new(5));

        assert_eq!(trie.node(child).symbol, SymbolId::new(5));
        assert_eq!(trie.node(child).count, 1);
        assert_eq!(trie.node(root).usage, 1);
        assert_eq!(trie.branch_count(root), 1);
    }

    #[test]
    fn add_child_increments_existing() {
        let mut trie = Trie::new();
        let root = trie.root();

        let first = trie.add_child(root, SymbolId::new(5));
        let second = trie.add_child(root, SymbolId::new(5));

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
        trie.add_child(root, SymbolId::new(10));
        trie.add_child(root, SymbolId::new(3));
        trie.add_child(root, SymbolId::new(7));
        trie.add_child(root, SymbolId::new(1));

        let children = trie.children(root);
        let symbols: Vec<SymbolId> = children.iter().map(|&r| trie.node(r).symbol).collect();

        assert_eq!(
            symbols,
            vec![
                SymbolId::new(1),
                SymbolId::new(3),
                SymbolId::new(7),
                SymbolId::new(10)
            ]
        );
    }

    #[test]
    fn find_child_existing() {
        let mut trie = Trie::new();
        let root = trie.root();
        let added = trie.add_child(root, SymbolId::new(42));

        let found = trie.find_child(root, SymbolId::new(42));
        assert_eq!(found, Some(added));
    }

    #[test]
    fn find_child_missing() {
        let trie = Trie::new();
        let root = trie.root();
        assert_eq!(trie.find_child(root, SymbolId::new(99)), None);
    }

    #[test]
    fn count_saturation_at_u16_max() {
        let mut trie = Trie::new();
        let root = trie.root();

        // Manually set count close to max.
        let child = trie.add_child(root, SymbolId::new(1)); // count = 1, usage = 1
        trie.nodes[child.as_usize()].count = u16::MAX - 1;
        trie.nodes[root.as_usize()].usage = u16::MAX as u32 - 1;

        // One more increment should work.
        trie.add_child(root, SymbolId::new(1));
        assert_eq!(trie.node(child).count, u16::MAX);
        assert_eq!(trie.node(root).usage, u16::MAX as u32);

        // Further increments should be silently dropped.
        trie.add_child(root, SymbolId::new(1));
        assert_eq!(trie.node(child).count, u16::MAX);
        assert_eq!(trie.node(root).usage, u16::MAX as u32);
    }

    #[test]
    fn multi_level_trie() {
        let mut trie = Trie::new();
        let root = trie.root();

        let level1 = trie.add_child(root, SymbolId::new(2));
        let level2 = trie.add_child(level1, SymbolId::new(3));
        let level3 = trie.add_child(level2, SymbolId::new(4));

        assert_eq!(trie.node(level3).symbol, SymbolId::new(4));
        assert_eq!(trie.node(level3).count, 1);
        assert_eq!(trie.node(level2).usage, 1);

        // Navigate back down.
        let found = trie.find_child(root, SymbolId::new(2)).unwrap();
        let found = trie.find_child(found, SymbolId::new(3)).unwrap();
        let found = trie.find_child(found, SymbolId::new(4)).unwrap();
        assert_eq!(found, level3);
    }

    #[test]
    fn trie_serde_roundtrip() {
        let mut trie = Trie::new();
        let root = trie.root();
        trie.add_child(root, SymbolId::new(2));
        trie.add_child(root, SymbolId::new(5));
        trie.add_child(root, SymbolId::new(2)); // increment count

        let json = serde_json::to_string(&trie).unwrap();
        let back: Trie = serde_json::from_str(&json).unwrap();

        let back_root = back.root();
        assert_eq!(back.branch_count(back_root), 2);
        assert_eq!(back.node(back_root).usage, 3);

        let child2 = back.find_child(back_root, SymbolId::new(2)).unwrap();
        assert_eq!(back.node(child2).count, 2);

        let child5 = back.find_child(back_root, SymbolId::new(5)).unwrap();
        assert_eq!(back.node(child5).count, 1);
    }

    #[test]
    fn len_and_is_empty() {
        let mut trie = Trie::new();
        assert_eq!(trie.len(), 1); // just root
        assert!(trie.is_empty());

        trie.add_child(trie.root(), SymbolId::new(1));
        assert_eq!(trie.len(), 2);
        assert!(!trie.is_empty());
    }

    #[test]
    fn default_creates_new_trie() {
        let trie = Trie::default();
        assert_eq!(trie.len(), 1);
        assert!(trie.is_empty());
    }

    #[test]
    fn usage_tracks_multiple_children() {
        let mut trie = Trie::new();
        let root = trie.root();

        trie.add_child(root, SymbolId::new(1));
        trie.add_child(root, SymbolId::new(2));
        trie.add_child(root, SymbolId::new(3));
        trie.add_child(root, SymbolId::new(1)); // increment existing

        assert_eq!(trie.node(root).usage, 4);
        assert_eq!(trie.branch_count(root), 3);
    }
}
