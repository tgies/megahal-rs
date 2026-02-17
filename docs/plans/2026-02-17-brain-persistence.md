# Brain Persistence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable saving and loading the MegaHAL brain (bidirectional Markov model) to/from disk so learned data persists across sessions.

**Architecture:** Add `serde::Serialize`/`Deserialize` derives to all model types across 5 crates (symbol-core, ngram-trie, symbol-dict, markov-chain, megahal). The megahal facade uses `bincode` 3 (with serde compat) to encode/decode the `BidirectionalModel<MegaHalSymbol>` to a binary file with a magic header. Only the model is serialized — runtime config (keywords, greetings, RNG) is not part of the brain.

**Tech Stack:** serde 1 (derive), bincode 3 (with `serde` feature for compat layer)

---

### Task 1: Add serde derives to symbol-core

**Files:**
- Modify: `crates/symbol-core/Cargo.toml`
- Modify: `crates/symbol-core/src/lib.rs`

**Step 1: Update Cargo.toml**

Add serde dependency:

```toml
[dependencies]
serde = { workspace = true }
```

**Step 2: Add derives to SymbolId**

In `crates/symbol-core/src/lib.rs`, add the import and derive:

```rust
use serde::{Serialize, Deserialize};
```

Change the `SymbolId` derive line to:

```rust
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
pub struct SymbolId(pub u16);
```

**Step 3: Write the failing test**

Add to the `tests` module in `crates/symbol-core/src/lib.rs`:

```rust
    #[test]
    fn symbol_id_serde_roundtrip() {
        let id = SymbolId(42);
        let json = serde_json::to_string(&id).unwrap();
        let back: SymbolId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }
```

This test requires `serde_json` as a dev-dependency. Add to `crates/symbol-core/Cargo.toml`:

```toml
[dev-dependencies]
serde_json = "1"
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p symbol-core symbol_id_serde_roundtrip`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/symbol-core/Cargo.toml crates/symbol-core/src/lib.rs
git commit -m "feat(symbol-core): add serde Serialize/Deserialize to SymbolId"
```

---

### Task 2: Add serde derives to ngram-trie

**Files:**
- Modify: `crates/ngram-trie/Cargo.toml`
- Modify: `crates/ngram-trie/src/lib.rs`

**Step 1: Update Cargo.toml**

```toml
[dependencies]
symbol-core.workspace = true
serde = { workspace = true }
```

**Step 2: Add derives to NodeRef, TrieNode, and Trie**

In `crates/ngram-trie/src/lib.rs`, add the import:

```rust
use serde::{Serialize, Deserialize};
```

Update the three struct derives:

```rust
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct NodeRef(u32);
```

```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct TrieNode {
    // ... fields unchanged
}
```

```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct Trie {
    nodes: Vec<TrieNode>,
}
```

Note: `TrieNode.children` is private but serde derives work on private fields since the generated code lives in the same module.

**Step 3: Write the failing test**

Add to the `tests` module in `crates/ngram-trie/src/lib.rs`:

```rust
    #[test]
    fn trie_serde_roundtrip() {
        let mut trie = Trie::new();
        let root = trie.root();
        trie.add_child(root, SymbolId(2));
        trie.add_child(root, SymbolId(5));
        trie.add_child(root, SymbolId(2)); // increment count

        let json = serde_json::to_string(&trie).unwrap();
        let back: Trie = serde_json::from_str(&json).unwrap();

        // Verify structure survived roundtrip.
        let back_root = back.root();
        assert_eq!(back.branch_count(back_root), 2);
        assert_eq!(back.node(back_root).usage, 3);

        let child2 = back.find_child(back_root, SymbolId(2)).unwrap();
        assert_eq!(back.node(child2).count, 2);

        let child5 = back.find_child(back_root, SymbolId(5)).unwrap();
        assert_eq!(back.node(child5).count, 1);
    }
```

Add to `crates/ngram-trie/Cargo.toml`:

```toml
[dev-dependencies]
serde_json = "1"
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p ngram-trie trie_serde_roundtrip`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/ngram-trie/Cargo.toml crates/ngram-trie/src/lib.rs
git commit -m "feat(ngram-trie): add serde Serialize/Deserialize to Trie, TrieNode, NodeRef"
```

---

### Task 3: Add serde derives to symbol-dict

**Files:**
- Modify: `crates/symbol-dict/Cargo.toml`
- Modify: `crates/symbol-dict/src/lib.rs`

**Step 1: Update Cargo.toml**

```toml
[dependencies]
symbol-core.workspace = true
serde = { workspace = true }
```

**Step 2: Add derives to SymbolDict**

In `crates/symbol-dict/src/lib.rs`, add the import:

```rust
use serde::{Serialize, Deserialize};
```

Add serde derives to SymbolDict. Since `SymbolDict<S>` is generic, serde needs `S: Serialize` and `S: Deserialize` bounds only when those traits are used, which serde's derive handles automatically with `#[serde(bound = "")]` or by default:

```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct SymbolDict<S: Symbol> {
    entries: Vec<S>,
    sorted_index: Vec<SymbolId>,
}
```

Note: serde's derive macro will automatically add `S: Serialize` for `Serialize` and `S: Deserialize<'de>` for `Deserialize`. Since `S: Symbol` is already a bound on the struct, and `Symbol` doesn't conflict, this should work. If the compiler complains about bounds, add `#[serde(bound = "S: Serialize + for<'a> Deserialize<'a>")]`.

**Step 3: Write the failing test**

Add to the `tests` module in `crates/symbol-dict/src/lib.rs`:

```rust
    #[test]
    fn dict_serde_roundtrip() {
        let mut dict = SymbolDict::<TestSym>::new();
        dict.intern(TestSym("HELLO".into()));
        dict.intern(TestSym("WORLD".into()));
        dict.intern(TestSym("APPLE".into()));

        let json = serde_json::to_string(&dict).unwrap();
        let back: SymbolDict<TestSym> = serde_json::from_str(&json).unwrap();

        assert_eq!(back.len(), dict.len());

        // Verify all symbols survived and are findable via sorted index.
        let id_hello = back.find(&TestSym("HELLO".into())).unwrap();
        assert_eq!(back.resolve(id_hello), &TestSym("HELLO".into()));

        let id_world = back.find(&TestSym("WORLD".into())).unwrap();
        assert_eq!(back.resolve(id_world), &TestSym("WORLD".into()));

        let id_apple = back.find(&TestSym("APPLE".into())).unwrap();
        assert_eq!(back.resolve(id_apple), &TestSym("APPLE".into()));
    }
```

This test also requires serde derives on TestSym. Update the test module's TestSym:

```rust
    #[derive(Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
    struct TestSym(String);
```

And add the test import + dev-dependency:

```rust
    use serde::{Serialize, Deserialize};
```

Add to `crates/symbol-dict/Cargo.toml`:

```toml
[dev-dependencies]
serde_json = "1"
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p symbol-dict dict_serde_roundtrip`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/symbol-dict/Cargo.toml crates/symbol-dict/src/lib.rs
git commit -m "feat(symbol-dict): add serde Serialize/Deserialize to SymbolDict"
```

---

### Task 4: Add serde derives to markov-chain

**Files:**
- Modify: `crates/markov-chain/Cargo.toml`
- Modify: `crates/markov-chain/src/lib.rs`

**Step 1: Update Cargo.toml**

```toml
[dependencies]
symbol-core.workspace = true
ngram-trie.workspace = true
symbol-dict.workspace = true
serde = { workspace = true }
```

**Step 2: Add derives to BidirectionalModel only**

In `crates/markov-chain/src/lib.rs`, add the import:

```rust
use serde::{Serialize, Deserialize};
```

Add derives to `BidirectionalModel` only (`ContextWindow` is transient state — never serialized):

```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct BidirectionalModel<S: Symbol> {
    // ... fields unchanged
}
```

**Step 3: Write the failing test**

Add to the `tests` module in `crates/markov-chain/src/lib.rs`:

```rust
    #[test]
    fn model_serde_roundtrip() {
        let mut model = BidirectionalModel::<TestSym>::new(2);
        model.learn(&make_tokens(&["A", "B", "C"]));
        model.learn(&make_tokens(&["X", "Y", "Z"]));

        let json = serde_json::to_string(&model).unwrap();
        let back: BidirectionalModel<TestSym> =
            serde_json::from_str(&json).unwrap();

        // Verify order preserved.
        assert_eq!(back.order, 2);

        // Verify dictionary preserved.
        assert_eq!(back.dictionary.len(), model.dictionary.len());
        assert!(back.dictionary.find(&TestSym("A".into())).is_some());
        assert!(back.dictionary.find(&TestSym("Z".into())).is_some());

        // Verify forward trie structure preserved.
        let root = back.forward.root();
        let id_a = back.dictionary.find(&TestSym("A".into())).unwrap();
        assert!(back.forward.find_child(root, id_a).is_some());

        // Verify backward trie structure preserved.
        let broot = back.backward.root();
        let id_c = back.dictionary.find(&TestSym("C".into())).unwrap();
        assert!(back.backward.find_child(broot, id_c).is_some());
    }
```

This requires serde derives on TestSym. Update:

```rust
    use serde::{Serialize, Deserialize};

    #[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
    struct TestSym(String);
```

Add to `crates/markov-chain/Cargo.toml`:

```toml
[dev-dependencies]
serde_json = "1"
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p markov-chain model_serde_roundtrip`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/markov-chain/Cargo.toml crates/markov-chain/src/lib.rs
git commit -m "feat(markov-chain): add serde Serialize/Deserialize to BidirectionalModel"
```

---

### Task 5: Add serde + bincode to megahal facade, implement save/load

**Files:**
- Modify: `Cargo.toml` (workspace root — update bincode to "3")
- Modify: `crates/megahal/Cargo.toml`
- Modify: `crates/megahal/src/lib.rs`

**Step 1: Update workspace bincode version**

In root `Cargo.toml`, update line 30:

```toml
bincode = { version = "3", features = ["serde"] }
```

**Step 2: Update megahal crate Cargo.toml**

Add serde and bincode dependencies:

```toml
[dependencies]
symbol-core.workspace = true
ngram-trie.workspace = true
symbol-dict.workspace = true
markov-chain.workspace = true
megahal-tokenizer.workspace = true
megahal-keywords.workspace = true
megahal-gen.workspace = true
rand.workspace = true
serde.workspace = true
bincode.workspace = true
```

**Step 3: Add Serialize/Deserialize to MegaHalSymbol**

In `crates/megahal/src/lib.rs`, add the import:

```rust
use serde::{Serialize, Deserialize};
```

Update the MegaHalSymbol derive:

```rust
#[derive(Clone, Debug, Hash, Serialize, Deserialize)]
pub struct MegaHalSymbol(Vec<u8>);
```

**Step 4: Write the failing test for save/load**

Add test to `crates/megahal/src/lib.rs`:

```rust
    #[test]
    fn save_load_brain_roundtrip() {
        let mut hal = trained_hal();
        let reply_before = hal.respond("Tell me about dogs.");

        let dir = std::env::temp_dir();
        let path = dir.join("megahal_test_brain.brn");
        hal.save_brain(&path).unwrap();

        // Create a fresh MegaHal and load the brain.
        let mut hal2 = test_hal();
        hal2.set_limit(GenerationLimit::Iterations(20));
        hal2.load_brain(&path).unwrap();

        // The loaded model should produce output (model has data).
        let reply_after = hal2.respond("Tell me about dogs.");
        assert!(!reply_after.is_empty());
        assert_ne!(reply_after, "I don't know enough to answer you yet!");

        // Verify dictionary size matches.
        assert_eq!(
            hal.model().dictionary.len(),
            hal2.model().dictionary.len()
        );

        fs::remove_file(&path).ok();
    }
```

Note: we can't assert `reply_before == reply_after` because the RNG state differs (hal already used its RNG, hal2 has a fresh seed). But we verify the loaded model has learned data.

**Step 5: Run test to verify it fails**

Run: `cargo test -p megahal save_load_brain_roundtrip`
Expected: FAIL — `save_brain` and `load_brain` methods don't exist yet

**Step 6: Implement save_brain and load_brain**

Add constants and methods to `crates/megahal/src/lib.rs`:

```rust
/// Magic bytes at the start of a brain file.
const BRAIN_MAGIC: &[u8; 8] = b"MHALRUST";

/// Brain file format version.
const BRAIN_VERSION: u8 = 1;
```

Add methods to `impl<R: Rng> MegaHal<R>`:

```rust
    /// Save the model to a binary brain file.
    ///
    /// The file format is: 8-byte magic ("MHALRUST") + 1-byte version + bincode-encoded model.
    /// Only the model (tries + dictionary) is saved — keyword config, greetings,
    /// generation limits, and RNG state are not included.
    pub fn save_brain(&self, path: &Path) -> io::Result<()> {
        let mut data = Vec::new();
        data.extend_from_slice(BRAIN_MAGIC);
        data.push(BRAIN_VERSION);

        let encoded = bincode::serde::encode_to_vec(&self.model, bincode::config::standard())
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        data.extend_from_slice(&encoded);

        fs::write(path, data)
    }

    /// Load a model from a binary brain file, replacing the current model.
    ///
    /// The model order is restored from the file. Keyword config, greetings,
    /// generation limits, and RNG are unaffected.
    pub fn load_brain(&mut self, path: &Path) -> io::Result<()> {
        let data = fs::read(path)?;

        if data.len() < 9 || &data[..8] != BRAIN_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a MegaHAL brain file",
            ));
        }
        if data[8] != BRAIN_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported brain version: {}", data[8]),
            ));
        }

        let (model, _len) =
            bincode::serde::decode_from_slice(&data[9..], bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        self.model = model;

        Ok(())
    }
```

**Step 7: Run test to verify it passes**

Run: `cargo test -p megahal save_load_brain_roundtrip`
Expected: PASS

**Step 8: Write additional edge-case tests**

```rust
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
        data.push(99); // unsupported version
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
        fs::write(&path, b"MHAL").unwrap(); // too short

        let mut hal = test_hal();
        let err = hal.load_brain(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        fs::remove_file(&path).ok();
    }
```

**Step 9: Run all tests to verify**

Run: `cargo test -p megahal`
Expected: All pass

**Step 10: Commit**

```bash
git add Cargo.toml crates/megahal/Cargo.toml crates/megahal/src/lib.rs
git commit -m "feat(megahal): implement brain persistence with save_brain/load_brain"
```

---

### Task 6: Add brain CLI args

**Files:**
- Modify: `src/main.rs`

**Step 1: Add --brain arg**

Add a new CLI arg to the `Args` struct:

```rust
    /// Brain file path. Loaded at startup (if exists), saved on exit.
    #[arg(long)]
    brain: Option<PathBuf>,
```

**Step 2: Load brain at startup**

After `hal.set_keyword_config(config)` and before the training file block, add brain loading:

```rust
    // Load brain if specified and file exists.
    if let Some(ref path) = args.brain {
        if path.exists() {
            eprintln!("Loading brain from {}...", path.display());
            hal.load_brain(path)?;
            eprintln!("Brain loaded.");
        }
    }
```

**Step 3: Save brain on exit**

After the conversation loop (after the `for line in stdin...` block), add:

```rust
    // Save brain on exit if path specified.
    if let Some(ref path) = args.brain {
        eprintln!("Saving brain to {}...", path.display());
        hal.save_brain(path)?;
        eprintln!("Brain saved.");
    }
```

**Step 4: Run manual smoke test**

```bash
# Train and save brain:
echo -e "Tell me about cats\nquit" | cargo run -- --train megahal.trn --data-dir megahal/ --seed 42 --max-iterations 20 --brain /tmp/test.brn

# Load brain without training file:
echo -e "Tell me about cats\nquit" | cargo run -- --seed 42 --max-iterations 20 --brain /tmp/test.brn
```

Expected: Second run should produce responses (not "I don't know enough") even without `--train`.

**Step 5: Commit**

```bash
git add src/main.rs
git commit -m "feat(cli): add --brain flag for brain persistence"
```

---

### Task 7: Run full test suite and verify

**Step 1: Run all tests**

Run: `cargo test --workspace`
Expected: All tests pass (109 existing + ~6 new serde roundtrip tests + ~3 brain persistence tests = ~118 total), zero warnings.

**Step 2: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: No warnings.

**Step 3: If issues, fix and re-run**

Fix any compilation errors, test failures, or clippy warnings before proceeding.

**Step 4: Final commit (if any fixes needed)**

```bash
git add -u
git commit -m "fix: address clippy/test issues from brain persistence"
```
