//! Profiling counters for tracking query execution frequencies.
//! Used with the `--show-time` flag to diagnose performance bottlenecks.

use std::sync::atomic::{AtomicU64, Ordering};

pub static TYPECHECK_SCC_CALLS: AtomicU64 = AtomicU64::new(0);
pub static TYPECHECK_DEP_GRAPH_CALLS: AtomicU64 = AtomicU64::new(0);
pub static FIND_IMPLICIT_CALLS: AtomicU64 = AtomicU64::new(0);
pub static GET_ACCUMULATED_CALLS: AtomicU64 = AtomicU64::new(0);

pub fn print_stats() {
    eprintln!("[counter] TypeCheckSCC calls:        {}", TYPECHECK_SCC_CALLS.load(Ordering::Relaxed));
    eprintln!("[counter] TypeCheckDepGraph calls:   {}", TYPECHECK_DEP_GRAPH_CALLS.load(Ordering::Relaxed));
    eprintln!("[counter] find_implicit calls:       {}", FIND_IMPLICIT_CALLS.load(Ordering::Relaxed));
    eprintln!("[counter] get_accumulated calls:     {}", GET_ACCUMULATED_CALLS.load(Ordering::Relaxed));
}
