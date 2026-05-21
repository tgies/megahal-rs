# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

Initial public release. Faithful Rust port of Jason Hutchens' 1998 MegaHAL chatbot.

### Added

- `MegaHal` engine generic over any `rand::Rng` PRNG, with configurable
  `GenerationLimit` (timeout, iteration cap, or both).
- `MegaHal::save_brain_to_writer` / `load_brain_from_reader` for in-memory
  brain serialization without filesystem I/O.
- `MegaHal::train_from_reader` for training from any `BufRead` source.
- `MegaHal::generate` - a no-learn variant of `respond` that returns
  `Option<String>` and does not feed the input back into the model.
- `MegaHal::set_fallback_reply` / `set_fallback_greeting` for replacing the
  hardcoded "I don't know enough..." and "Hello!" messages.
- `MegaHalError` thiserror enum surfaced from the new reader/writer APIs.
- Workspace-internal crates: `symbol-core`, `ngram-trie`, `symbol-dict`,
  `markov-chain`, `megahal-tokenizer`, `megahal-keywords`, `megahal-gen`.
  These crates are published as transitive dependencies of `megahal`; their
  APIs are not stable and consumers should depend on the `megahal` facade.
- `megahal-cli` crate with `cargo install megahal-cli` for the `megahal` binary.

### Fixed

- `capitalize` no longer upper-cases the letter after a `!`/`.`/`?` when no
  whitespace follows (e.g. `a.b.c` now correctly produces `A.b.c`).
- Reply generation no longer echoes the user's input when that was the only
  thing the model could produce; falls back to the canned reply instead.
- The backward generation phase now runs even when the forward phase produces
  no tokens, matching the C reference behavior.
- `babble` no longer panics when a context node's `usage` count is zero;
  treats that case as sentence-terminating.
