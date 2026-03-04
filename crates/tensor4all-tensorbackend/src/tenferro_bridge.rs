//! Runtime bridge for tenferro backend execution.
//!
//! This module centralizes runtime selection so backend code stays
//! implementation-agnostic (CPU today, GPU-ready extension point).

use anyhow::{anyhow, Result};
use std::env;
use tenferro_prims::{CpuBackend, CpuContext};

/// Runtime kind for tenferro execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeKind {
    /// CPU runtime.
    Cpu,
    /// CUDA runtime (reserved).
    Cuda,
    /// ROCm runtime (reserved).
    Rocm,
}

/// Active tenferro prims backend used by tensor4all.
///
/// This alias keeps backend selection localized to this bridge module.
pub(crate) type ActivePrimsBackend = CpuBackend;

fn parse_runtime_kind() -> RuntimeKind {
    match env::var("T4A_TENFERRO_RUNTIME") {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "cpu" => RuntimeKind::Cpu,
            "cuda" => RuntimeKind::Cuda,
            "rocm" => RuntimeKind::Rocm,
            _ => RuntimeKind::Cpu,
        },
        Err(_) => RuntimeKind::Cpu,
    }
}

fn cpu_threads() -> usize {
    let parsed = env::var("T4A_TENFERRO_CPU_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    parsed.max(1)
}

/// Run a tenferro op against currently selected runtime.
///
/// Current implementation executes on CPU and returns explicit errors for GPU
/// runtime requests until tenferro GPU runtime wiring is enabled in tensor4all.
pub fn with_tenferro_ctx<R>(
    op: &'static str,
    f: impl FnOnce(&mut CpuContext) -> Result<R>,
) -> Result<R> {
    match parse_runtime_kind() {
        RuntimeKind::Cpu => {
            let mut ctx = CpuContext::new(cpu_threads());
            f(&mut ctx)
        }
        RuntimeKind::Cuda => Err(anyhow!(
            "{}: CUDA runtime is not yet wired in tensor4all tenferro backend",
            op
        )),
        RuntimeKind::Rocm => Err(anyhow!(
            "{}: ROCm runtime is not yet wired in tensor4all tenferro backend",
            op
        )),
    }
}
