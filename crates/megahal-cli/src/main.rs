//! MegaHAL CLI: an interactive conversational chatbot.
//!
//! Thin wrapper over the `megahal` library crate.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use megahal::{GenerationLimit, KeywordConfig, MegaHal, SwapTable, load_swap_file, load_word_list};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// MegaHAL: a conversational chatbot using bidirectional Markov chains.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Model order (trie depth). Default: 5.
    #[arg(long, default_value_t = 5)]
    order: u8,

    /// PRNG seed for reproducible output.
    #[arg(long)]
    seed: Option<u64>,

    /// Training file path.
    #[arg(long)]
    train: Option<PathBuf>,

    /// Brain file path. Loaded at startup if it exists, saved on exit.
    /// Defaults to $XDG_DATA_HOME/megahal/megahal.brn (platform equivalent
    /// on macOS / Windows). Pass --no-brain to disable persistence.
    #[arg(long, conflicts_with = "no_brain")]
    brain: Option<PathBuf>,

    /// Disable brain persistence: no file is loaded at startup or saved on exit.
    #[arg(long)]
    no_brain: bool,

    /// Directory containing support files (megahal.ban, .aux, .grt, .swp).
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Generation timeout in milliseconds.
    #[arg(long, default_value_t = 1000)]
    timeout_ms: u64,

    /// Maximum generation iterations (0 = no limit).
    #[arg(long, default_value_t = 0)]
    max_iterations: usize,
}

/// Default brain location: `$XDG_DATA_HOME/megahal/megahal.brn` on Linux,
/// `~/Library/Application Support/megahal/megahal.brn` on macOS,
/// `%APPDATA%\megahal\megahal.brn` on Windows.
fn default_brain_path() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("megahal").join("megahal.brn"))
}

/// Resolve the effective brain path from the parsed args.
///
/// Returns `None` if `--no-brain` is set or no default path is available
/// (which can happen on platforms where `dirs::data_dir()` returns `None`).
fn resolve_brain_path(args: &Args) -> Option<PathBuf> {
    if args.no_brain {
        return None;
    }
    args.brain.clone().or_else(default_brain_path)
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let seed = args.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });

    let rng = SmallRng::seed_from_u64(seed);
    let mut hal = MegaHal::new(args.order, rng);

    // Set generation limit.
    let limit = match (args.timeout_ms, args.max_iterations) {
        (ms, 0) => GenerationLimit::Timeout(std::time::Duration::from_millis(ms)),
        (0, n) => GenerationLimit::Iterations(n),
        (ms, n) => GenerationLimit::Both {
            timeout: std::time::Duration::from_millis(ms),
            max_iterations: n,
        },
    };
    hal.set_limit(limit);

    // Load support files if data directory specified.
    if let Some(ref dir) = args.data_dir {
        let mut config = KeywordConfig::default();

        if let Ok(words) = load_word_list(&dir.join("megahal.ban")) {
            config.banned = words.into_iter().collect();
        }
        if let Ok(words) = load_word_list(&dir.join("megahal.aux")) {
            config.auxiliary = words.into_iter().collect();
        }
        if let Ok(pairs) = load_swap_file(&dir.join("megahal.swp")) {
            config.swap = SwapTable { pairs };
        }
        if let Ok(words) = load_word_list(&dir.join("megahal.grt")) {
            hal.set_greetings(words);
        }

        hal.set_keyword_config(config);
    }

    let brain_path = resolve_brain_path(&args);

    // Load brain if its file exists.
    if let Some(ref path) = brain_path
        && path.exists()
    {
        eprintln!("Loading brain from {}...", path.display());
        hal.load_brain(path)?;
        eprintln!("Brain loaded.");
    }

    // Train from file if specified.
    if let Some(ref path) = args.train {
        eprintln!("Training from {}...", path.display());
        hal.train_from_file(path)?;
        eprintln!("Training complete.");
    }

    // Initial greeting.
    let greeting = hal.greet();
    println!("MegaHAL: {greeting}");

    // Conversation loop.
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("quit") || trimmed.eq_ignore_ascii_case("exit") {
            break;
        }

        let reply = hal.respond(trimmed);
        writeln!(stdout, "MegaHAL: {reply}")?;
        stdout.flush()?;
    }

    // Save brain on exit unless --no-brain was set.
    if let Some(ref path) = brain_path {
        ensure_parent_dir(path)?;
        eprintln!("Saving brain to {}...", path.display());
        hal.save_brain(path)?;
        eprintln!("Brain saved.");
    }

    Ok(())
}

fn ensure_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
