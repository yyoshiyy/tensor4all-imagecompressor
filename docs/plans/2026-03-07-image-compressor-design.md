# Image Compressor Design

**Date:** 2026-03-07
**Crate:** `tensor4all-imagecompressor`

## Overview

A new Rust crate that compresses images using Quantics Tensor Train (QTT) via
`tensor4all-quanticstci`. The compressed representation can be reconstructed at
arbitrary resolution ("infinite zoom"), making it suitable for ML image
datasets (PNG/JPEG, RGB).

## Architecture

### Processing Flow

```
PNG/JPEG input
    ↓ image crate
Split into R, G, B channels as 2D f64 arrays [0.0, 1.0]
    ↓
Compress each channel independently with quanticstci
  f_R(x, y), f_G(x, y), f_B(x, y) → SimpleTT<f64>
    ↓
CompressedImage { tt_r, tt_g, tt_b, bits, original_size }
    ↓ reconstruct(width, height)
Evaluate TTs at scaled coordinates → RgbImage
    ↓ image crate
PNG/JPEG output
```

### Core Types

```rust
pub struct CompressedImage {
    tt_r: SimpleTT<f64>,
    tt_g: SimpleTT<f64>,
    tt_b: SimpleTT<f64>,
    bits: usize,           // grid size = 2^bits x 2^bits
    original_width: u32,
    original_height: u32,
}

pub struct CompressOptions {
    pub tolerance: f64,
    pub max_rank: Option<usize>,
}
```

### Public API

```rust
pub fn compress(path: &Path, opts: CompressOptions) -> Result<CompressedImage>;

impl CompressedImage {
    pub fn reconstruct(&self, width: u32, height: u32) -> Result<RgbImage>;
    pub fn compression_ratio(&self) -> f64;
}
```

## Quantics Grid

For an image of size `W x H`:
- `bits = ceil(log2(max(W, H)))`
- Grid: `2^bits x 2^bits` (padded with zeros if not power-of-2)
- Each channel is treated as `f(x, y)` over integer grid points

For reconstruction at `(W', H')`:
- Scale coordinates: `x_grid = x * (2^bits - 1) / (W' - 1)`
- Evaluate QTT at nearest integer grid points (nearest-neighbor)

## Dependencies

- `tensor4all-quanticstci` — QTT compression via TCI
- `tensor4all-simplett` — `SimpleTT` type for evaluation
- `image` — PNG/JPEG I/O

## Error Handling

- `thiserror`-based `ImageCompressorError` enum
- Variants: `IoError`, `ImageError`, `CompressionError`

## Testing

1. **Solid color image** — round-trip compression reproduces pixel values exactly
2. **Round-trip accuracy** — `compress → reconstruct(original_size)` pixel error within `tolerance`
3. **Upscale** — `reconstruct(2 * original_size)` succeeds and returns correct dimensions
4. **Compression ratio** — compressed size is smaller than original for natural images

## Future Extensions

- YCbCr color space for perceptual quality improvement
- Python bindings via `tensor4all-capi`
- Serialization of `CompressedImage` (HDF5 or binary)
