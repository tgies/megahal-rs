# megahal-rs

[![CI](https://github.com/tgies/megahal-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/tgies/megahal-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/megahal.svg)](https://crates.io/crates/megahal)
[![Docs.rs](https://docs.rs/megahal/badge.svg)](https://docs.rs/megahal)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust reimplementation of [MegaHAL](https://en.wikipedia.org/wiki/MegaHAL),
the 1998 bidirectional Markov chain chatbot by Jason Hutchens.

Faithful to the original C algorithm (tokenization, keyword extraction,
bidirectional generation, surprise evaluation), organized as a modular Cargo
workspace.

## Install

```bash
# Library (embed the engine in your own program). `rand` is needed because
# `MegaHal` is generic over any `rand::Rng` and you supply the PRNG.
cargo add megahal rand

# CLI (interactive chatbot at the terminal, installs as `megahal`)
cargo install megahal-cli
```

## Library usage

```rust
use megahal::{MegaHal, GenerationLimit};
use rand::{SeedableRng, rngs::SmallRng};
use std::time::Duration;

let mut bot = MegaHal::new(5, SmallRng::seed_from_u64(0xC0FFEE));

// Per-frame budget for a game HUD: a few ms, capped iterations.
bot.set_limit(GenerationLimit::Both {
    timeout: Duration::from_millis(4),
    max_iterations: 8,
});

// Train from your own corpus (any &str or BufRead).
for line in include_str!("../assets/lore.txt").lines() {
    bot.learn(line);
}

// `respond` learns from the input first, then replies.
let reply = bot.respond("system status");

// `generate` is a no-learn variant for cases where the input is untrusted
// and shouldn't pollute the model. Returns None on empty replies so the
// caller can supply its own placeholder.
let chatter: Option<String> = bot.generate("hello");
```

The brain file format is **not** compatible with the original C MegaHAL's
`.brn` files. See [`MegaHal::save_brain_to_writer`] and
[`load_brain_from_reader`] for in-memory persistence without filesystem I/O.

## Run the CLI

```bash
# Train on the bundled corpus and start chatting
megahal --train data/megahal.trn --data-dir data

# Reproducible output with a fixed seed and no persistence
megahal --seed 42 --train data/megahal.trn --no-brain
```

Type `quit` or `exit` to stop.

The CLI persists its model between runs at
`$XDG_DATA_HOME/megahal/megahal.brn` (`~/Library/Application Support/megahal/megahal.brn`
on macOS, `%APPDATA%\megahal\megahal.brn` on Windows). Override the path with
`--brain PATH` or disable persistence entirely with `--no-brain`.

## Architecture

```
megahal-cli          thin CLI wrapper (clap)
  └── megahal        facade: MegaHalSymbol, config, brain persistence
        ├── megahal-gen        reply generation, babble, surprise scoring
        │   ├── megahal-keywords   keyword extraction, swap/ban/aux tables
        │   │   └── megahal-markov ── ngram-trie ── symbol-core
        │   └── megahal-markov
        ├── megahal-tokenizer  text tokenization (MegaHAL boundary rules)
        └── megahal-markov     bidirectional model + context window
              ├── ngram-trie   arena-based n-gram frequency trie
              │   └── symbol-core  Symbol trait + SymbolId
              └── symbol-dict  generic interning dictionary
                  └── symbol-core
```

The lower seven crates are workspace-internal: they are published to crates.io
because cargo requires it for `megahal` itself to publish, but their APIs are
not stable and consumers should depend on the `megahal` facade.

## Test

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo llvm-cov --workspace      # line coverage (requires cargo-llvm-cov)
```

## Credits

Original 1998 C implementation by Jason Hutchens. The MegaHAL algorithm is
described in his 1998 paper "Introducing MegaHAL" and in the source of
[MegaHALv8](https://github.com/kranzky/megahal).

## License

MIT
