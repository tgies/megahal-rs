//! CLI integration tests for the `megahal` binary.

use std::path::{Path, PathBuf};

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

/// Path to the bundled MegaHAL data directory.
fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("data")
}

/// Default command helper.
fn megahal_cmd() -> Command {
    let mut cmd = cargo_bin_cmd!("megahal");
    cmd.arg("--no-brain");
    cmd
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
    // Default greeting when untrained.
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
    // Empty stdin.
    megahal_cmd()
        .args(["--seed", "42", "--max-iterations", "10"])
        .write_stdin("")
        .assert()
        .success();
}

#[test]
fn empty_lines_are_skipped() {
    // Empty lines are ignored.
    megahal_cmd()
        .args(["--seed", "42", "--max-iterations", "10"])
        .write_stdin("\n\n\nquit\n")
        .assert()
        .success()
        // Only the greeting is printed.
        .stdout(predicate::function(|output: &str| {
            output.matches("MegaHAL:").count() == 1
        }));
}

#[test]
fn responds_to_input() {
    // Train and respond.
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
        // Greeting + response.
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
    // Verify data-dir loads support files.
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
}

// ---------------------------------------------------------------------------
// Brain persistence
// ---------------------------------------------------------------------------

#[test]
fn brain_save_and_load() {
    let dir = std::env::temp_dir();
    let brain_path = dir.join("megahal_cli_test_brain.brn");

    // Clean up previous files.
    let _ = std::fs::remove_file(&brain_path);

    // Train and save the brain.
    cargo_bin_cmd!("megahal")
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

    assert!(brain_path.exists(), "brain file should have been created");
    let size = std::fs::metadata(&brain_path).unwrap().len();
    assert!(
        size > 100,
        "brain file should be non-trivial, got {size} bytes"
    );

    // Load saved brain.
    cargo_bin_cmd!("megahal")
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
        // Model is loaded.
        .stdout(predicate::function(|output: &str| {
            output.matches("MegaHAL:").count() >= 2
        }));

    let _ = std::fs::remove_file(&brain_path);
}

#[test]
fn brain_and_no_brain_conflict() {
    cargo_bin_cmd!("megahal")
        .args(["--brain", "/tmp/x.brn", "--no-brain"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot be used with"));
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
