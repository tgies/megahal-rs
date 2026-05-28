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
// `convert` subcommand: import C MegaHAL `MegaHALv8` brains
// ---------------------------------------------------------------------------

/// Build a minimal but well-formed `MegaHALv8` brain (cookie, order, two
/// empty roots, dictionary with the two sentinels). Suitable for testing
/// the conversion pipeline without needing a real C MegaHAL fixture.
fn write_minimal_v8_brain(path: &Path) {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"MegaHALv8");
    buf.push(5); // order
    // Forward root: symbol=0, usage=0, count=0, branch=0.
    buf.extend_from_slice(&[0; 10]);
    // Backward root: identical.
    buf.extend_from_slice(&[0; 10]);
    // Dictionary count = 2.
    buf.extend_from_slice(&2u32.to_le_bytes());
    // "<ERROR>"
    buf.push(7);
    buf.extend_from_slice(b"<ERROR>");
    // "<FIN>"
    buf.push(5);
    buf.extend_from_slice(b"<FIN>");
    std::fs::write(path, buf).unwrap();
}

#[test]
fn convert_v8_brain_writes_native_format() {
    let dir = std::env::temp_dir();
    let source = dir.join("megahal_cli_convert_src.brn");
    let dest = dir.join("megahal_cli_convert_dst.brn");
    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&dest);

    write_minimal_v8_brain(&source);

    cargo_bin_cmd!("megahal")
        .args(["convert", source.to_str().unwrap(), dest.to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::contains("Loading V8 brain"))
        .stderr(predicate::str::contains("Writing converted brain"));

    let written = std::fs::read(&dest).unwrap();
    assert!(
        written.starts_with(b"MHALRUST"),
        "converted brain should start with MHALRUST magic"
    );

    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&dest);
}

#[test]
fn convert_missing_source_fails() {
    cargo_bin_cmd!("megahal")
        .args(["convert", "/nonexistent/megahal.brn", "/tmp/out.brn"])
        .assert()
        .failure();
}

#[test]
fn convert_rejects_non_v8_input() {
    let dir = std::env::temp_dir();
    let source = dir.join("megahal_cli_convert_bad.brn");
    let dest = dir.join("megahal_cli_convert_bad_out.brn");
    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&dest);

    std::fs::write(&source, b"NotaBrainAtAll").unwrap();

    cargo_bin_cmd!("megahal")
        .args(["convert", source.to_str().unwrap(), dest.to_str().unwrap()])
        .assert()
        .failure();
    assert!(!dest.exists(), "no output should be written on failure");

    let _ = std::fs::remove_file(&source);
}

#[test]
fn convert_does_not_start_chat_loop() {
    let dir = std::env::temp_dir();
    let source = dir.join("megahal_cli_convert_quiet_src.brn");
    let dest = dir.join("megahal_cli_convert_quiet_dst.brn");
    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&dest);

    write_minimal_v8_brain(&source);

    // No stdin needed; the subcommand should exit without prompting.
    cargo_bin_cmd!("megahal")
        .args(["convert", source.to_str().unwrap(), dest.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::is_empty());

    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&dest);
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
