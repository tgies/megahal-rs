//! CLI integration tests for the `megahal` binary.
//!
//! Uses `assert_cmd` to spawn the binary as a subprocess, pipe stdin,
//! and assert on stdout/stderr/exit code.

use std::path::{Path, PathBuf};

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

/// Path to the MegaHAL data directory (bundled in the repo).
fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data")
}

fn megahal_cmd() -> Command {
    Command::from(cargo_bin_cmd!("megahal"))
}

// ---------------------------------------------------------------------------
// Basic CLI behavior
// ---------------------------------------------------------------------------

#[test]
fn help_flag() {
    megahal_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("bidirectional Markov chains"));
}

#[test]
fn version_flag() {
    megahal_cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("megahal-cli"));
}

// ---------------------------------------------------------------------------
// Conversation loop
// ---------------------------------------------------------------------------

#[test]
fn greeting_on_startup() {
    // With no training, greeting is "Hello!".
    megahal_cmd()
        .args(["--seed", "42", "--max-iterations", "10"])
        .write_stdin("quit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("MegaHAL: Hello!"));
}

#[test]
fn quit_exits_cleanly() {
    megahal_cmd()
        .args(["--seed", "42", "--max-iterations", "10"])
        .write_stdin("quit\n")
        .assert()
        .success();
}

#[test]
fn exit_exits_cleanly() {
    megahal_cmd()
        .args(["--seed", "42", "--max-iterations", "10"])
        .write_stdin("exit\n")
        .assert()
        .success();
}

#[test]
fn eof_exits_cleanly() {
    // Empty stdin (EOF immediately after greeting).
    megahal_cmd()
        .args(["--seed", "42", "--max-iterations", "10"])
        .write_stdin("")
        .assert()
        .success();
}

#[test]
fn empty_lines_are_skipped() {
    // Empty lines should not produce responses; only the actual input should.
    megahal_cmd()
        .args(["--seed", "42", "--max-iterations", "10"])
        .write_stdin("\n\n\nquit\n")
        .assert()
        .success()
        // Should have exactly one "MegaHAL:" line (the greeting), not four.
        .stdout(predicate::function(|output: &str| {
            output.matches("MegaHAL:").count() == 1
        }));
}

#[test]
fn responds_to_input() {
    // Train on a file so the model has enough data to generate replies.
    megahal_cmd()
        .args([
            "--seed",
            "42",
            "--max-iterations",
            "20",
            "--train",
            data_dir().join("megahal.trn").to_str().unwrap(),
        ])
        .write_stdin("Tell me about the world.\nquit\n")
        .assert()
        .success()
        // Greeting + at least one response = at least 2 "MegaHAL:" lines.
        .stdout(predicate::function(|output: &str| {
            output.matches("MegaHAL:").count() >= 2
        }));
}

#[test]
fn case_insensitive_quit() {
    megahal_cmd()
        .args(["--seed", "42", "--max-iterations", "10"])
        .write_stdin("QUIT\n")
        .assert()
        .success();
}

#[test]
fn case_insensitive_exit() {
    megahal_cmd()
        .args(["--seed", "42", "--max-iterations", "10"])
        .write_stdin("Exit\n")
        .assert()
        .success();
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

#[test]
fn train_flag_loads_file() {
    megahal_cmd()
        .args([
            "--seed",
            "42",
            "--max-iterations",
            "10",
            "--train",
            data_dir().join("megahal.trn").to_str().unwrap(),
        ])
        .write_stdin("quit\n")
        .assert()
        .success()
        .stderr(predicate::str::contains("Training from"))
        .stderr(predicate::str::contains("Training complete"));
}

#[test]
fn train_missing_file_fails() {
    megahal_cmd()
        .args(["--seed", "42", "--train", "/nonexistent/path/megahal.trn"])
        .write_stdin("quit\n")
        .assert()
        .failure();
}

// ---------------------------------------------------------------------------
// Data directory
// ---------------------------------------------------------------------------

#[test]
fn data_dir_loads_support_files() {
    // When --data-dir is given with real files, the greeting should use
    // training data + greeting keywords rather than the bare "Hello!".
    megahal_cmd()
        .args([
            "--seed",
            "42",
            "--max-iterations",
            "20",
            "--train",
            data_dir().join("megahal.trn").to_str().unwrap(),
            "--data-dir",
            data_dir().to_str().unwrap(),
        ])
        .write_stdin("quit\n")
        .assert()
        .success();
    // We don't assert a specific greeting since it depends on the model,
    // but the process should complete without error.
}

// ---------------------------------------------------------------------------
// Brain persistence
// ---------------------------------------------------------------------------

#[test]
fn brain_save_and_load() {
    let dir = std::env::temp_dir();
    let brain_path = dir.join("megahal_cli_test_brain.brn");

    // Remove any leftover file from a previous run.
    let _ = std::fs::remove_file(&brain_path);

    // Step 1: Train and save brain.
    megahal_cmd()
        .args([
            "--seed",
            "42",
            "--max-iterations",
            "10",
            "--train",
            data_dir().join("megahal.trn").to_str().unwrap(),
            "--brain",
            brain_path.to_str().unwrap(),
        ])
        .write_stdin("quit\n")
        .assert()
        .success()
        .stderr(predicate::str::contains("Saving brain"))
        .stderr(predicate::str::contains("Brain saved"));

    // Brain file should exist.
    assert!(brain_path.exists(), "brain file should have been created");
    let size = std::fs::metadata(&brain_path).unwrap().len();
    assert!(
        size > 100,
        "brain file should be non-trivial, got {size} bytes"
    );

    // Step 2: Load brain without training.
    megahal_cmd()
        .args([
            "--seed",
            "42",
            "--max-iterations",
            "20",
            "--brain",
            brain_path.to_str().unwrap(),
        ])
        .write_stdin("Tell me something.\nquit\n")
        .assert()
        .success()
        .stderr(predicate::str::contains("Loading brain"))
        .stderr(predicate::str::contains("Brain loaded"))
        // Should produce a real response (not fallback), since model is loaded.
        .stdout(predicate::function(|output: &str| {
            output.matches("MegaHAL:").count() >= 2
        }));

    let _ = std::fs::remove_file(&brain_path);
}

// ---------------------------------------------------------------------------
// Deterministic output with --seed
// ---------------------------------------------------------------------------

#[test]
fn seed_produces_deterministic_output() {
    let run = || {
        megahal_cmd()
            .args([
                "--seed",
                "123",
                "--max-iterations",
                "20",
                "--train",
                data_dir().join("megahal.trn").to_str().unwrap(),
            ])
            .write_stdin("Tell me about computers.\nquit\n")
            .output()
            .expect("should run")
    };

    let out1 = run();
    let out2 = run();

    assert_eq!(
        out1.stdout, out2.stdout,
        "same seed should produce identical stdout"
    );
}
