# megahal-rs

A Rust reimplementation of [MegaHAL](https://en.wikipedia.org/wiki/MegaHAL), the 1998 bidirectional Markov chain chatbot by Jason Hutchens.

Faithful to the original C algorithm (tokenization, keyword extraction, bidirectional generation, surprise evaluation) but organized as a modular Cargo workspace with generic, reusable crates at the foundation.

## Architecture

```
megahal-cli          thin CLI wrapper (clap)
  └── megahal        facade: MegaHalSymbol, config, brain persistence
        ├── megahal-gen        reply generation, babble, surprise scoring
        │   ├── megahal-keywords   keyword extraction, swap/ban/aux tables
        │   │   └── markov-chain ── ngram-trie ── symbol-core
        │   └── markov-chain
        ├── megahal-tokenizer  text tokenization (MegaHAL boundary rules)
        └── markov-chain       bidirectional model + context window
              ├── ngram-trie   arena-based n-gram frequency trie
              │   └── symbol-core  Symbol trait + SymbolId
              └── symbol-dict  generic interning dictionary
                  └── symbol-core
```

The bottom four crates (`symbol-core`, `ngram-trie`, `symbol-dict`, `markov-chain`) are fully generic over any `Symbol` type — no string or MegaHAL assumptions.

## Build & Run

```bash
cargo build --release

# Train on the bundled corpus and start chatting
cargo run --release -- \
  --train data/megahal.trn \
  --data-dir data \
  --brain brain.bin

# Reproducible output with a fixed seed
cargo run --release -- --seed 42 --train data/megahal.trn
```

Type `quit` or `exit` to stop. If `--brain` is set, the model is saved on exit and reloaded on next start.

## Test

```bash
cargo test --workspace          # 160+ unit, integration, and CLI tests
cargo llvm-cov --workspace      # line coverage (requires cargo-llvm-cov)
cargo clippy --workspace        # lint check
```

## License

MIT
