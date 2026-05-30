//! Read C MegaHAL `MegaHALv8` brain files.
//!
//! C MegaHAL's `.brn` is a self-contained binary file: cookie + order +
//! forward trie + backward trie + embedded dictionary. The dictionary that
//! ships separately as `megahal.dic` is an output-only debug dump and is not
//! consumed at load time.
//!
//! ## Format
//!
//! ```text
//! 9 bytes   "MegaHALv8" cookie
//! 1 byte    model order (u8)
//! ...       forward trie (depth-first; see node layout below)
//! ...       backward trie (depth-first; same node layout)
//! 4 bytes   dictionary count (u32 LE; 8 bytes in 8-byte mode)
//! N times   1-byte length + UTF-8 bytes per entry
//! ```
//!
//! Each trie node:
//!
//! ```text
//! 2 bytes   symbol id (u16 LE)
//! 4 bytes   usage (u32 LE; 8 bytes in 8-byte mode)
//! 2 bytes   count (u16 LE)
//! 2 bytes   branch count (u16 LE)
//! ...       branch_count child nodes recursively
//! ```
//!
//! ## 4-byte vs 8-byte usage width
//!
//! C MegaHAL stores `usage` as `unsigned long`, which is 4 bytes on ILP32 and
//! 8 bytes on LP64 (`megahal/megahal.c` line 142). We replicate the
//! megahal-js heuristic (`megahal-js/src/binary.js` lines 264-276) for
//! auto-detection: peek the root's `branch` field at offset 18 (4-byte
//! assumption) and offset 22 (8-byte assumption); if offset-18 reads 0 and
//! offset-22 reads >0, treat the file as 8-byte. The same width applies to
//! the dictionary count word.
//!
//! ## Endianness
//!
//! C MegaHAL writes in native byte order. This importer assumes little-endian
//! source files (every commodity platform in practical use). Brains written
//! on a big-endian host are not supported.
//!
//! ## Sentinels
//!
//! The C dictionary's first two entries are `<ERROR>` and `<FIN>`, matching
//! [`SymbolDict::new()`]'s pre-seeded entries. We use [`SymbolDict::intern`]
//! for every entry in file order: the sentinels resolve to existing IDs 0
//! and 1; user symbols land at IDs >= 2.

use std::io::{self, Read};

use megahal_markov::BidirectionalModel;
use ngram_trie::{NodeRef, Trie, TrieNode};
use symbol_core::SymbolId;
use symbol_dict::SymbolDict;

use crate::{MegaHalError, MegaHalSymbol};

const V8_COOKIE: &[u8; 9] = b"MegaHALv8";

/// Maximum dictionary size, dictated by the 16-bit symbol ID space.
const MAX_DICT_ENTRIES: usize = 65536;

/// Parse a C MegaHAL `MegaHALv8` brain into a [`BidirectionalModel`].
///
/// Reads the entire input into memory (brains are small in practice and the
/// recursive trie format is awkward to stream without `Seek`).
pub(crate) fn parse_v8_brain<R: Read>(
    mut reader: R,
) -> Result<BidirectionalModel<MegaHalSymbol>, MegaHalError> {
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;

    let mut cursor = Cursor::new(&buf);

    let cookie = cursor.read_array::<9>()?;
    if &cookie != V8_COOKIE {
        return Err(MegaHalError::BadV8Cookie);
    }
    let order = cursor.read_u8()?;

    let width = detect_usage_width(&buf);

    let mut forward_nodes = Vec::new();
    read_node(&mut cursor, &mut forward_nodes, width)?;

    let mut backward_nodes = Vec::new();
    read_node(&mut cursor, &mut backward_nodes, width)?;

    let dict_count = cursor.read_count(width)?;
    if dict_count > MAX_DICT_ENTRIES {
        return Err(MegaHalError::V8DictionaryTooLarge(dict_count));
    }

    let mut dictionary: SymbolDict<MegaHalSymbol> = SymbolDict::new();
    for _ in 0..dict_count {
        let len = cursor.read_u8()? as usize;
        let bytes = cursor.read_bytes(len)?;
        // C load_word reads raw bytes with no encoding validation
        // (megahal.c:1384). Intern as raw bytes, applying ASCII uppercasing
        // to match C's convention (toupper in C locale, megahal.c:965-970).
        dictionary.intern(MegaHalSymbol::from_raw_bytes(bytes));
    }

    Ok(BidirectionalModel {
        order,
        forward: Trie::from_raw_nodes(forward_nodes),
        backward: Trie::from_raw_nodes(backward_nodes),
        dictionary,
    })
}

/// Heuristic from megahal-js (`binary.js:264-276`): inspect the root node's
/// branch count under both width assumptions; only switch to 8-byte mode if
/// the 4-byte position is 0 and the 8-byte position is nonzero.
fn detect_usage_width(buf: &[u8]) -> UsageWidth {
    if buf.len() < 24 {
        return UsageWidth::Four;
    }
    let branch_if_four = u16::from_le_bytes([buf[18], buf[19]]);
    let branch_if_eight = u16::from_le_bytes([buf[22], buf[23]]);
    if branch_if_four == 0 && branch_if_eight > 0 {
        UsageWidth::Eight
    } else {
        UsageWidth::Four
    }
}

#[derive(Copy, Clone, Debug)]
enum UsageWidth {
    Four,
    Eight,
}

fn read_node(
    cursor: &mut Cursor<'_>,
    nodes: &mut Vec<TrieNode>,
    width: UsageWidth,
) -> Result<NodeRef, MegaHalError> {
    let symbol_raw = cursor.read_u16()?;
    let usage = cursor.read_count_u32(width)?;
    let count = cursor.read_u16()?;
    let branch = cursor.read_u16()? as usize;

    // Reserve our slot in the arena before recursing so children get larger
    // NodeRefs and the parent's index stays stable.
    let my_idx = nodes.len();
    nodes.push(TrieNode::from_raw(
        SymbolId::new(symbol_raw),
        usage,
        count,
        Vec::with_capacity(branch),
    ));

    let mut children = Vec::with_capacity(branch);
    for _ in 0..branch {
        children.push(read_node(cursor, nodes, width)?);
    }

    // Replace the placeholder children list now that the recursive walk has
    // populated the arena and we know the child NodeRefs.
    let symbol = nodes[my_idx].symbol;
    let usage = nodes[my_idx].usage;
    let count = nodes[my_idx].count;
    nodes[my_idx] = TrieNode::from_raw(symbol, usage, count, children);

    Ok(NodeRef::from_usize(my_idx))
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], MegaHalError> {
        let end = self.pos.checked_add(n).ok_or_else(|| {
            MegaHalError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "V8 brain: read length overflow",
            ))
        })?;
        if end > self.buf.len() {
            return Err(MegaHalError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "V8 brain: unexpected end of input",
            )));
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], MegaHalError> {
        let slice = self.read_bytes(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(slice);
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8, MegaHalError> {
        Ok(self.read_array::<1>()?[0])
    }

    fn read_u16(&mut self) -> Result<u16, MegaHalError> {
        Ok(u16::from_le_bytes(self.read_array::<2>()?))
    }

    fn read_u32(&mut self) -> Result<u32, MegaHalError> {
        Ok(u32::from_le_bytes(self.read_array::<4>()?))
    }

    /// Read a width-prefixed count, used for `usage` fields. The high 4 bytes
    /// are discarded in 8-byte mode (C MegaHAL's `usage` is a `BYTE4` widened
    /// to `unsigned long`, capped at u32 for our purposes).
    fn read_count_u32(&mut self, width: UsageWidth) -> Result<u32, MegaHalError> {
        let low = self.read_u32()?;
        if matches!(width, UsageWidth::Eight) {
            let _high = self.read_u32()?;
        }
        Ok(low)
    }

    /// Read the dictionary count, which uses the same width as `usage`.
    fn read_count(&mut self, width: UsageWidth) -> Result<usize, MegaHalError> {
        let low = self.read_u32()? as usize;
        if matches!(width, UsageWidth::Eight) {
            let _high = self.read_u32()?;
        }
        Ok(low)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Specification for a V8 trie node used by the test serializer.
    struct NodeSpec {
        symbol: u16,
        usage: u32,
        count: u16,
        children: Vec<NodeSpec>,
    }

    /// Build a V8 brain stream from a tree spec.
    ///
    /// `eight_byte` controls whether `usage` and dictionary count fields are
    /// padded to 8 bytes (matching brains written on LP64 hosts).
    fn build_v8(
        order: u8,
        forward: NodeSpec,
        backward: NodeSpec,
        dict_entries: &[&str],
        eight_byte: bool,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(V8_COOKIE);
        out.push(order);
        write_node(&mut out, &forward, eight_byte);
        write_node(&mut out, &backward, eight_byte);
        out.extend_from_slice(&(dict_entries.len() as u32).to_le_bytes());
        if eight_byte {
            out.extend_from_slice(&0u32.to_le_bytes());
        }
        for entry in dict_entries {
            let bytes = entry.as_bytes();
            assert!(bytes.len() <= u8::MAX as usize);
            out.push(bytes.len() as u8);
            out.extend_from_slice(bytes);
        }
        out
    }

    fn write_node(out: &mut Vec<u8>, node: &NodeSpec, eight_byte: bool) {
        out.extend_from_slice(&node.symbol.to_le_bytes());
        out.extend_from_slice(&node.usage.to_le_bytes());
        if eight_byte {
            out.extend_from_slice(&0u32.to_le_bytes());
        }
        out.extend_from_slice(&node.count.to_le_bytes());
        out.extend_from_slice(&(node.children.len() as u16).to_le_bytes());
        for child in &node.children {
            write_node(out, child, eight_byte);
        }
    }

    fn empty_root() -> NodeSpec {
        NodeSpec {
            symbol: 0,
            usage: 0,
            count: 0,
            children: Vec::new(),
        }
    }

    /// Tree shape: root -> [symbol 2 -> [symbol 3], symbol 4]. Forces branching
    /// and multi-level recursion, exercising the depth-first reader and the
    /// children-sorted invariant.
    fn sample_tree() -> NodeSpec {
        NodeSpec {
            symbol: 0,
            usage: 5,
            count: 0,
            children: vec![
                NodeSpec {
                    symbol: 2,
                    usage: 3,
                    count: 3,
                    children: vec![NodeSpec {
                        symbol: 3,
                        usage: 0,
                        count: 3,
                        children: Vec::new(),
                    }],
                },
                NodeSpec {
                    symbol: 4,
                    usage: 0,
                    count: 2,
                    children: Vec::new(),
                },
            ],
        }
    }

    #[test]
    fn minimal_brain_loads() {
        let buf = build_v8(
            5,
            empty_root(),
            empty_root(),
            &["<ERROR>", "<FIN>", "HELLO"],
            false,
        );
        let model = parse_v8_brain(&buf[..]).unwrap();
        assert_eq!(model.order, 5);
        assert_eq!(model.dictionary.len(), 3);
        assert!(model.forward.is_empty());
        assert!(model.backward.is_empty());
    }

    #[test]
    fn populated_tries_load() {
        let buf = build_v8(
            5,
            sample_tree(),
            sample_tree(),
            &["<ERROR>", "<FIN>", "A", "B", "C"],
            false,
        );
        let model = parse_v8_brain(&buf[..]).unwrap();
        assert_eq!(model.dictionary.len(), 5);

        let root = model.forward.root();
        assert_eq!(model.forward.node(root).usage, 5);
        let children = model.forward.children(root);
        assert_eq!(children.len(), 2);
        let first_child_symbol = model.forward.node(children[0]).symbol.as_u16();
        let second_child_symbol = model.forward.node(children[1]).symbol.as_u16();
        assert_eq!(first_child_symbol, 2);
        assert_eq!(second_child_symbol, 4);
    }

    #[test]
    fn bad_cookie_rejected() {
        let mut buf = b"NotaBrain".to_vec();
        buf.push(5);
        match parse_v8_brain(&buf[..]) {
            Err(MegaHalError::BadV8Cookie) => {}
            other => panic!("expected BadV8Cookie, got {other:?}"),
        }
    }

    #[test]
    fn truncated_returns_io_error() {
        // Cookie only; missing order byte and everything after.
        let buf = V8_COOKIE.to_vec();
        match parse_v8_brain(&buf[..]) {
            Err(MegaHalError::Io(e)) => assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof),
            other => panic!("expected Io(UnexpectedEof), got {other:?}"),
        }
    }

    #[test]
    fn dictionary_too_large_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(V8_COOKIE);
        buf.push(5);
        // Two empty roots.
        write_node(&mut buf, &empty_root(), false);
        write_node(&mut buf, &empty_root(), false);
        // Claim a dictionary larger than MAX_DICT_ENTRIES; no entries follow.
        buf.extend_from_slice(&((MAX_DICT_ENTRIES as u32) + 1).to_le_bytes());
        match parse_v8_brain(&buf[..]) {
            Err(MegaHalError::V8DictionaryTooLarge(n)) => assert!(n > MAX_DICT_ENTRIES),
            other => panic!("expected V8DictionaryTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn four_byte_and_eight_byte_load_equivalently() {
        let dict = ["<ERROR>", "<FIN>", "A", "B", "C"];
        let four = build_v8(5, sample_tree(), sample_tree(), &dict, false);
        let eight = build_v8(5, sample_tree(), sample_tree(), &dict, true);

        let m4 = parse_v8_brain(&four[..]).unwrap();
        let m8 = parse_v8_brain(&eight[..]).unwrap();

        assert_eq!(m4.order, m8.order);
        assert_eq!(m4.dictionary.len(), m8.dictionary.len());
        assert_eq!(m4.forward.len(), m8.forward.len());
        assert_eq!(m4.backward.len(), m8.backward.len());
        assert_eq!(
            m4.forward.node(m4.forward.root()).usage,
            m8.forward.node(m8.forward.root()).usage,
        );
    }

    #[test]
    fn width_heuristic_selects_eight_byte() {
        // Build an 8-byte brain whose root has branches; the detector must
        // pick 8-byte mode and parse cleanly.
        let buf = build_v8(
            5,
            sample_tree(),
            sample_tree(),
            &["<ERROR>", "<FIN>", "A", "B", "C"],
            true,
        );
        assert!(matches!(detect_usage_width(&buf), UsageWidth::Eight));
        let model = parse_v8_brain(&buf[..]).unwrap();
        assert_eq!(model.forward.children(model.forward.root()).len(), 2);
    }

    #[test]
    fn width_heuristic_selects_four_byte() {
        let buf = build_v8(
            5,
            sample_tree(),
            sample_tree(),
            &["<ERROR>", "<FIN>"],
            false,
        );
        assert!(matches!(detect_usage_width(&buf), UsageWidth::Four));
    }

    /// Build a V8 brain stream with raw-byte dictionary entries (no UTF-8 req).
    fn build_v8_raw_dict(order: u8, dict_entries: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(V8_COOKIE);
        out.push(order);
        // Two empty roots.
        write_node(&mut out, &empty_root(), false);
        write_node(&mut out, &empty_root(), false);
        out.extend_from_slice(&(dict_entries.len() as u32).to_le_bytes());
        for entry in dict_entries {
            assert!(entry.len() <= u8::MAX as usize);
            out.push(entry.len() as u8);
            out.extend_from_slice(entry);
        }
        out
    }

    // Dictionary words with non-UTF-8 bytes (e.g. 0xFF) must load without error
    // and the stored bytes must be preserved (with ASCII uppercasing applied).
    #[test]
    fn non_utf8_dictionary_entry_loads_and_preserves_raw_bytes() {
        // 0xFF is not valid UTF-8, so the loader must intern it as raw bytes.
        let non_utf8_word: &[u8] = &[b'W', b'O', b'R', b'D', 0xFF];
        let buf = build_v8_raw_dict(5, &[b"<ERROR>", b"<FIN>", non_utf8_word]);
        let model = parse_v8_brain(&buf[..]).expect("must load non-UTF-8 bytes without error");
        assert_eq!(model.dictionary.len(), 3);

        // The word with 0xFF should be in the dictionary.  Resolve entry at
        // index 2 (IDs: 0=<ERROR>, 1=<FIN>, 2=the non-UTF-8 word).
        use symbol_core::SymbolId;
        let sym = model.dictionary.resolve(SymbolId::new(2));
        // ASCII bytes uppercased, 0xFF unchanged.
        assert_eq!(sym.as_bytes(), &[b'W', b'O', b'R', b'D', 0xFF]);
    }
}
