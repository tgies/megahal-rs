# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0]

First release on crates.io. Renames the inner bidirectional Markov crate from `markov-chain` to `megahal-markov` to avoid a name conflict, exposes new APIs in the facade, and fixes four algorithm bugs relative to the C reference.

### Added

- `MegaHal::save_brain_to_writer` / `load_brain_from_reader` for
  in-memory brain serialization without filesystem I/O.
- `MegaHal::train_from_reader` for training from any `BufRead` source.
- `MegaHal::generate`: a no-learn variant of `respond`.
- `MegaHal::set_fallback_reply` / `set_fallback_greeting` for replacing
  the hardcoded "I don't know enough..." and "Hello!" messages.
- `MegaHalError` for reader and writer errors.
- `megahal-cli` crate with `cargo install megahal-cli` for the `megahal`
  binary, extracted from the workspace root.
- Runnable examples under `crates/megahal/examples`: `chatbot` (stdin
  loop), `game_hud` (no-learn `generate` with a tight iteration budget),
  and `brain_persist` (save/load via reader/writer).

### Changed

- Renamed the `markov-chain` crate to `megahal-markov`. The name
  `markov-chain` is held on crates.io by an unrelated 2017 crate.
- `SymbolId`'s tuple-struct field is no longer public; construct via
  `SymbolId::new(value)`. The `symbol-core` crate's API is documented as
  unstable; consumers should depend on the `megahal` facade.
- Expanded the `megahal` crate-level rustdoc with an API tour, concepts
  section, brain-file format note, and a recipe for shipping a
  pre-trained brain via `include_bytes!`.

### Fixed

- `capitalize` no longer upper-cases the letter after a `!`/`.`/`?` when
  no whitespace follows (e.g. `a.b.c` now correctly produces `A.b.c`).
- Reply generation no longer echoes the user's input when that was the
  only thing the model could produce; falls back to the canned reply
  instead.
- The backward generation phase now runs even when the forward phase
  produces no tokens, matching the C reference behavior.
- `babble` no longer panics when a context node's `usage` count is zero;
  treats that case as sentence-terminating.

## [0.1.0]

GitHub-only release. Initial public tag of the Rust port of Jason
Hutchens' 1998 MegaHAL chatbot. Not published to crates.io.

### Added

- `MegaHal` engine generic over any `rand::Rng` PRNG, with configurable
  `GenerationLimit` (timeout, iteration cap, or both).
- Workspace-internal crates: `symbol-core`, `ngram-trie`, `symbol-dict`,
  `markov-chain` (later renamed `megahal-markov`), `megahal-tokenizer`,
  `megahal-keywords`, `megahal-gen`.
