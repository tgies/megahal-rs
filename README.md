# megahal-rs

[![CI](https://github.com/tgies/megahal-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/tgies/megahal-rs/actions/workflows/ci.yml)
[![Mutation Testing](https://github.com/tgies/megahal-rs/actions/workflows/mutants-full.yml/badge.svg?branch=main)](https://github.com/tgies/megahal-rs/actions/workflows/mutants-full.yml)
[![Mutation Tested](https://img.shields.io/badge/mutation--tested-cargo--mutants-blueviolet?logo=rust)](https://mutants.rs/)
[![Crates.io](https://img.shields.io/crates/v/megahal.svg)](https://crates.io/crates/megahal)
[![Docs.rs](https://docs.rs/megahal/badge.svg)](https://docs.rs/megahal)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


A Rust reimplementation of [MegaHAL](https://en.wikipedia.org/wiki/MegaHAL),
the 1998 bidirectional Markov chain chatbot by Jason Hutchens.

Faithful to the original C algorithm (tokenization, keyword extraction,
bidirectional generation, surprise evaluation).

Provided as a CLI and as a library.

## Install

### CLI

```bash
# CLI (interactive chatbot at the terminal, installs as `megahal`)
cargo install megahal-cli
megahal
```

### As a library

```bash
# Note: We use a pluggable RNG rather than wiring one in
cargo add megahal rand
```

## Library usage

```rust
use megahal::{MegaHal, GenerationLimit};
use rand::{SeedableRng, rngs::SmallRng};
use std::time::Duration;

let mut bot = MegaHal::new(5, SmallRng::seed_from_u64(42));

// Limit generation timeout and iterations.
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

// `generate` is a no-learn variant. Returns `None` on empty replies.
let reply: Option<String> = bot.generate("hello");
```

The brain file format is not compatible with the original C MegaHAL's
`.brn` files. See the `save_brain_to_writer` and `load_brain_from_reader`
methods for in-memory persistence without filesystem I/O.

## Run the CLI

If running from a local repository checkout, you can train on the bundled sample corpus:
```bash
megahal --train data/megahal.trn --data-dir data
```

Otherwise, train on your own corpus text file:
```bash
megahal --train path/to/corpus.txt
```

To run with a fixed seed and disable brain file persistence:
```bash
megahal --seed 42 --train path/to/corpus.txt --no-brain
```

Type `quit` or `exit` to stop.

The CLI persists its model between runs at
`$XDG_DATA_HOME/megahal/megahal.brn` (`~/Library/Application Support/megahal/megahal.brn`
on macOS, `%APPDATA%\megahal\megahal.brn` on Windows). Override the path with
`--brain PATH` or disable persistence entirely with `--no-brain`.

## Architecture

```
megahal-cli                        thin CLI wrapper
  └── megahal                      facade: MegaHalSymbol, config, brain persistence
        ├── megahal-gen            reply generation, babble, surprise scoring
        │   ├── megahal-keywords   keyword extraction, swap/ban/aux tables
        │   │   └── megahal-markov ── ngram-trie ── symbol-core
        │   └── megahal-markov
        ├── megahal-tokenizer      text tokenization (MegaHAL boundary rules)
        └── megahal-markov         bidirectional model + context window
              ├── ngram-trie       arena-based n-gram frequency trie
              │   └── symbol-core  Symbol trait + SymbolId
              └── symbol-dict      generic interning dictionary
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

### Mutation Testing

This repository uses [`cargo-mutants`](https://mutants.rs/) to verify test quality by introducing small modifications (mutants) to the source code and checking if the test suite catches them.

To run mutation testing locally:
```bash
cargo mutants --workspace
```

## Credits

Original 1998 C implementation by Jason Hutchens. The MegaHAL algorithm is
described in his 1998 paper "Introducing MegaHAL" and in the source of
[MegaHALv8](https://github.com/kranzky/megahal).

## References

- Hutchens, Jason L.; Alder, Michael D. (1998), ["Introducing MegaHAL"][introducing-megahal], NeMLaP3/CoNLL98 Workshop on Human-Computer Conversation, ACL, pp. 271--274.
- Hutchens, Jason L. (1997), ["How to Pass the Turing Test by Cheating"][turing-test-cheating], Technical Report TR97-05, Department of E&E Engineering, University of Western Australia.
- Hutchens, Jason L., ["How MegaHAL Works"][how-megahal-works], MegaHAL homepage.
- Hutchens, Jason L., [Original MegaHAL C source code][megahal-github], GitHub.

[introducing-megahal]: https://aclanthology.org/W98-1233.pdf
[turing-test-cheating]: https://courses.cs.umbc.edu/471/papers/hutchens.pdf
[how-megahal-works]: https://megahal.sourceforge.net/How.html
[megahal-github]: https://github.com/pteichman/megahal

## License

MIT
