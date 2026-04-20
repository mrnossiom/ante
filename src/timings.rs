//! Helpers for the `--show-time` CLI flag. Each timed compiler phase calls
//! [`time_phase`] so every line is printed in a consistent column layout and
//! the running total accumulates into [`TOTAL_PHASE_TIME`] for the footer.

use std::cell::Cell;
use std::time::{Duration, Instant};

/// Width of the label column in `--show-time` output; sized to fit the
/// longest phase label we print (e.g. "Name resolution", "Object emission").
pub const PHASE_LABEL_WIDTH: usize = 18;

thread_local! {
    /// Accumulates elapsed time from every `time_phase` call so the footer
    /// prints a Total that actually sums what we measured.
    pub static TOTAL_PHASE_TIME: Cell<Duration> = const { Cell::new(Duration::ZERO) };
}

/// Run `f`, returning its result. If `show_time` is set, time the call and
/// print one phase-timing line in the shared `--show-time` format.
///
/// This must be called from the main thread
pub fn time_phase<T>(label: &str, show_time: bool, f: impl FnOnce() -> T) -> T {
    if !show_time {
        return f();
    }
    let t = Instant::now();
    let result = f();
    let elapsed = t.elapsed();
    TOTAL_PHASE_TIME.with(|cell| cell.set(cell.get() + elapsed));
    eprintln!("  {:<PHASE_LABEL_WIDTH$}  {:>9.2} ms", label, elapsed.as_secs_f64() * 1000.0);
    result
}

/// This must be called from the main thread
pub fn print_total_time_of_phases() {
    let total = TOTAL_PHASE_TIME.with(|cell| cell.get());
    eprintln!("  {:-<width$}", "", width = PHASE_LABEL_WIDTH + 2 + 9 + 3);
    eprintln!("  {:<PHASE_LABEL_WIDTH$}  {:>9.2} ms", "Total", total.as_secs_f64() * 1000.0);
}
