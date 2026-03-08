# Image Compressor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a `tensor4all-imagecompressor` crate that compresses PNG/JPEG images into QTT (Quantics Tensor Train) format and reconstructs them at arbitrary resolution.

**Architecture:** Each RGB channel is independently compressed as a 2D discrete function `f(x, y)` using `quanticscrossinterpolate_discrete`. The image is padded to a `2^bits x 2^bits` grid. Reconstruction evaluates the TT at scaled coordinates for any target resolution.

**Tech Stack:** `tensor4all-quanticstci`, `tensor4all-simplett`, `image` crate (PNG/JPEG I/O), `thiserror`, `anyhow`

---

### Task 1: Create the crate skeleton

**Files:**
- Create: `crates/tensor4all-imagecompressor/Cargo.toml`
- Create: `crates/tensor4all-imagecompressor/src/lib.rs`
- Modify: `Cargo.toml` (workspace root)

**Step 1: Add `image` to workspace dependencies**

In `Cargo.toml` (root), add under `[workspace.dependencies]`:
```toml
image = "0.25"
```

**Step 2: Create `crates/tensor4all-imagecompressor/Cargo.toml`**

```toml
[package]
name = "tensor4all-imagecompressor"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true

[dependencies]
tensor4all-quanticstci = { path = "../tensor4all-quanticstci" }
tensor4all-simplett = { path = "../tensor4all-simplett" }
anyhow = { workspace = true }
thiserror = { workspace = true }
image = { workspace = true }
```

**Step 3: Create `crates/tensor4all-imagecompressor/src/lib.rs`**

```rust
//! Image compression using Quantics Tensor Train (QTT).
//!
//! Compresses PNG/JPEG images into QTT format and reconstructs
//! them at arbitrary resolution.

mod compress;
mod error;
mod reconstruct;

pub use compress::{compress, CompressOptions, CompressedImage};
pub use error::ImageCompressorError;
```

**Step 4: Add crate to workspace in root `Cargo.toml`**

In `[workspace] members`, add:
```toml
"crates/tensor4all-imagecompressor",
```

**Step 5: Verify it compiles (no src files yet — will fail but check for syntax)**

```bash
cd /path/to/tensor4all-rs
cargo check -p tensor4all-imagecompressor 2>&1 | head -20
```

Expected: errors about missing modules `compress`, `error`, `reconstruct` — that's fine at this stage.

**Step 6: Commit**

```bash
git add crates/tensor4all-imagecompressor/ Cargo.toml
git commit -m "feat(imagecompressor): add crate skeleton"
```

---

### Task 2: Error type

**Files:**
- Create: `crates/tensor4all-imagecompressor/src/error.rs`

**Step 1: Write the failing test**

Add to `crates/tensor4all-imagecompressor/src/lib.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = ImageCompressorError::InvalidImageSize { width: 0, height: 0 };
        assert!(e.to_string().contains("0"));
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo nextest run --release -p tensor4all-imagecompressor 2>&1 | tail -10
```
Expected: compile error — `ImageCompressorError` not found.

**Step 3: Create `crates/tensor4all-imagecompressor/src/error.rs`**

```rust
use thiserror::Error;

/// Errors from the image compressor.
#[derive(Debug, Error)]
pub enum ImageCompressorError {
    #[error("Invalid image size: {width}x{height}")]
    InvalidImageSize { width: u32, height: u32 },

    #[error("Image I/O error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("Compression failed: {0}")]
    CompressionError(#[from] anyhow::Error),
}
```

**Step 4: Run test to verify it passes**

```bash
cargo nextest run --release -p tensor4all-imagecompressor 2>&1 | tail -10
```
Expected: `test_error_display` PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-imagecompressor/src/
git commit -m "feat(imagecompressor): add error type"
```

---

### Task 3: Grid utilities (padding to power-of-2)

**Files:**
- Create: `crates/tensor4all-imagecompressor/src/compress.rs` (partial)

**Background:** `quanticscrossinterpolate_discrete` requires size to be a power of 2, and both dimensions must be the same. We need `bits = ceil(log2(max(W, H)))`, grid size = `2^bits`.

**Step 1: Write the failing test for `bits_for_size`**

Add a `#[cfg(test)]` module in `compress.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_for_size() {
        assert_eq!(bits_for_size(1, 1), 1);   // 2^1 = 2 >= max(1,1)=1, but min is 1
        assert_eq!(bits_for_size(4, 4), 2);   // 2^2 = 4
        assert_eq!(bits_for_size(5, 3), 3);   // 2^3 = 8 >= 5
        assert_eq!(bits_for_size(256, 128), 8); // 2^8 = 256
        assert_eq!(bits_for_size(257, 100), 9); // 2^9 = 512 >= 257
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cargo nextest run --release -p tensor4all-imagecompressor test_bits_for_size 2>&1 | tail -10
```
Expected: compile error.

**Step 3: Write minimal implementation**

Create `crates/tensor4all-imagecompressor/src/compress.rs`:
```rust
use anyhow::Result;
use image::{DynamicImage, RgbImage};
use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};
use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};

use crate::error::ImageCompressorError;

/// Options for image compression.
pub struct CompressOptions {
    /// Relative tolerance for QTT compression (lower = more accurate, larger TT).
    pub tolerance: f64,
    /// Maximum bond dimension (None = unlimited).
    pub max_rank: Option<usize>,
}

impl Default for CompressOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-3,
            max_rank: None,
        }
    }
}

/// A compressed image stored as three QTTs (one per RGB channel).
pub struct CompressedImage {
    /// QTT for red channel.
    pub(crate) tt_r: TensorTrain<f64>,
    /// QTT for green channel.
    pub(crate) tt_g: TensorTrain<f64>,
    /// QTT for blue channel.
    pub(crate) tt_b: TensorTrain<f64>,
    /// Number of bits per axis (grid size = 2^bits x 2^bits).
    pub(crate) bits: usize,
    /// Original image width.
    pub(crate) original_width: u32,
    /// Original image height.
    pub(crate) original_height: u32,
}

/// Compute the number of bits needed so that 2^bits >= max(w, h).
/// Minimum is 1 (grid size 2).
pub(crate) fn bits_for_size(w: u32, h: u32) -> usize {
    let max_dim = w.max(h) as usize;
    let bits = (max_dim as f64).log2().ceil() as usize;
    bits.max(1)
}

/// Compress a PNG/JPEG image file into a CompressedImage.
pub fn compress(
    img: &DynamicImage,
    opts: CompressOptions,
) -> Result<CompressedImage, ImageCompressorError> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    if w == 0 || h == 0 {
        return Err(ImageCompressorError::InvalidImageSize { width: w, height: h });
    }

    let bits = bits_for_size(w, h);
    let grid_size = 1usize << bits; // 2^bits
    let sizes = vec![grid_size, grid_size];

    // Build pixel lookup from the original image.
    // Coordinates are 0-indexed. Out-of-bounds pixels are 0.0.
    let pixels: Vec<u8> = rgb.into_raw(); // RGBRGB... row-major
    let w = w as usize;
    let h = h as usize;

    let get_channel = |channel: usize| {
        let pixels_ref = pixels.clone();
        move |idx: &[i64]| -> f64 {
            // idx is 1-indexed (quanticsgrids convention)
            let x = (idx[0] - 1) as usize; // column
            let y = (idx[1] - 1) as usize; // row
            if x < w && y < h {
                pixels_ref[(y * w + x) * 3 + channel] as f64 / 255.0
            } else {
                0.0
            }
        }
    };

    let mut qtci_opts = QtciOptions::default().with_tolerance(opts.tolerance);
    if let Some(max_rank) = opts.max_rank {
        qtci_opts = qtci_opts.with_maxbonddim(max_rank);
    }

    let compress_channel = |channel: usize| -> Result<TensorTrain<f64>, ImageCompressorError> {
        let f = get_channel(channel);
        let (qtci, _, _) =
            quanticscrossinterpolate_discrete(&sizes, f, None, qtci_opts.clone())
                .map_err(ImageCompressorError::CompressionError)?;
        qtci.tensor_train()
            .map_err(ImageCompressorError::CompressionError)
    };

    let tt_r = compress_channel(0)?;
    let tt_g = compress_channel(1)?;
    let tt_b = compress_channel(2)?;

    Ok(CompressedImage {
        tt_r,
        tt_g,
        tt_b,
        bits,
        original_width: w as u32,
        original_height: h as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits_for_size() {
        assert_eq!(bits_for_size(1, 1), 1);
        assert_eq!(bits_for_size(4, 4), 2);
        assert_eq!(bits_for_size(5, 3), 3);
        assert_eq!(bits_for_size(256, 128), 8);
        assert_eq!(bits_for_size(257, 100), 9);
    }
}
```

**Step 4: Run test to verify it passes**

```bash
cargo nextest run --release -p tensor4all-imagecompressor test_bits_for_size 2>&1 | tail -10
```
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-imagecompressor/src/
git commit -m "feat(imagecompressor): add compress function and grid utilities"
```

---

### Task 4: Reconstruct function

**Files:**
- Create: `crates/tensor4all-imagecompressor/src/reconstruct.rs`
- Modify: `crates/tensor4all-imagecompressor/src/compress.rs` (add `reconstruction_ratio`, `reconstruct` to `CompressedImage`)

**Background:** To reconstruct at `(W', H')`, scale each output pixel `(x', y')` to the QTT grid:
- `x_grid = round(x' * (grid_size - 1) / (W' - 1))` (clamped to `[0, grid_size-1]`)
- Evaluate TT at `[x_grid, y_grid]` (0-indexed for TT, but `quantics_tci evaluate` uses 1-indexed grid indices via the grid object — however we extracted the raw `TensorTrain`, so we use the quantics index directly)

**Important:** `TensorTrain::evaluate` takes a `&[usize]` of quantics indices (0-indexed). The quantics indices for a 2D discrete grid with `Interleaved` scheme interleave bits of x and y. We must convert grid (x, y) → quantics index manually, OR use `qtci.evaluate(&[i64])` (1-indexed) before extracting the TT.

**Simpler approach:** Keep the raw `TensorTrain` plus a helper that converts `(x_grid, y_grid)` → quantics index using the `InherentDiscreteGrid`.

**Revised `CompressedImage` fields:** Also store `InherentDiscreteGrid` for coordinate conversion.

**Step 1: Update `CompressedImage` to store grid**

In `compress.rs`, update `CompressedImage`:
```rust
use quanticsgrids::InherentDiscreteGrid;

pub struct CompressedImage {
    pub(crate) tt_r: TensorTrain<f64>,
    pub(crate) tt_g: TensorTrain<f64>,
    pub(crate) tt_b: TensorTrain<f64>,
    pub(crate) grid: InherentDiscreteGrid,
    pub(crate) bits: usize,
    pub(crate) original_width: u32,
    pub(crate) original_height: u32,
}
```

Update `compress` to build and store the grid:
```rust
use quanticsgrids::{InherentDiscreteGrid, UnfoldingScheme};

// After computing bits/grid_size, build the shared grid:
let grid = InherentDiscreteGrid::builder(&vec![bits, bits])
    .with_unfolding_scheme(UnfoldingScheme::Interleaved)
    .build()
    .map_err(|e| ImageCompressorError::CompressionError(anyhow::anyhow!(e)))?;
```

Also store `grid: grid.clone()` in the returned `CompressedImage`.

**Step 2: Write the failing test for `reconstruct`**

Add to `crates/tensor4all-imagecompressor/src/lib.rs` tests:
```rust
#[test]
fn test_reconstruct_solid_color() {
    use image::{DynamicImage, RgbImage, Rgb};
    // 4x4 solid red image
    let mut img_buf = RgbImage::new(4, 4);
    for pixel in img_buf.pixels_mut() {
        *pixel = Rgb([255u8, 0, 0]);
    }
    let dyn_img = DynamicImage::ImageRgb8(img_buf);
    let opts = CompressOptions { tolerance: 1e-6, max_rank: None };
    let compressed = compress(&dyn_img, opts).unwrap();
    let reconstructed = compressed.reconstruct(4, 4).unwrap();
    for pixel in reconstructed.pixels() {
        assert!(pixel[0] > 200, "red channel should be high");
        assert!(pixel[1] < 10, "green channel should be near zero");
        assert!(pixel[2] < 10, "blue channel should be near zero");
    }
}
```

**Step 3: Run test to verify it fails**

```bash
cargo nextest run --release -p tensor4all-imagecompressor test_reconstruct_solid_color 2>&1 | tail -10
```
Expected: compile error — `reconstruct` not found.

**Step 4: Create `crates/tensor4all-imagecompressor/src/reconstruct.rs`**

```rust
use image::RgbImage;
use tensor4all_simplett::AbstractTensorTrain;

use crate::compress::CompressedImage;
use crate::error::ImageCompressorError;

impl CompressedImage {
    /// Reconstruct the image at the specified resolution.
    ///
    /// The output can be larger or smaller than the original image.
    pub fn reconstruct(&self, width: u32, height: u32) -> Result<RgbImage, ImageCompressorError> {
        if width == 0 || height == 0 {
            return Err(ImageCompressorError::InvalidImageSize { width, height });
        }

        let grid_size = (1usize << self.bits) as f64;
        let w = width as usize;
        let h = height as usize;

        let mut pixels = vec![0u8; w * h * 3];

        for py in 0..h {
            for px in 0..w {
                // Scale output pixel to QTT grid coordinates (0-indexed)
                let gx = if w > 1 {
                    ((px as f64) * (grid_size - 1.0) / (w as f64 - 1.0)).round() as usize
                } else {
                    0
                };
                let gy = if h > 1 {
                    ((py as f64) * (grid_size - 1.0) / (h as f64 - 1.0)).round() as usize
                } else {
                    0
                };
                let gx = gx.min((1usize << self.bits) - 1);
                let gy = gy.min((1usize << self.bits) - 1);

                // Convert (gx, gy) grid indices to quantics index using the stored grid.
                // InherentDiscreteGrid uses 1-indexed grid indices.
                let grid_idx = vec![(gx as i64) + 1, (gy as i64) + 1];
                let quantics_idx = self
                    .grid
                    .grididx_to_quantics(&grid_idx)
                    .map_err(|e| ImageCompressorError::CompressionError(anyhow::anyhow!(e)))?;
                // Convert 1-indexed quantics to 0-indexed for TensorTrain::evaluate
                let q_usize: Vec<usize> = quantics_idx
                    .iter()
                    .map(|&x| (x - 1) as usize)
                    .collect();

                let r = self
                    .tt_r
                    .evaluate(&q_usize)
                    .map_err(|e| ImageCompressorError::CompressionError(anyhow::anyhow!(e)))?
                    .clamp(0.0, 1.0);
                let g = self
                    .tt_g
                    .evaluate(&q_usize)
                    .map_err(|e| ImageCompressorError::CompressionError(anyhow::anyhow!(e)))?
                    .clamp(0.0, 1.0);
                let b = self
                    .tt_b
                    .evaluate(&q_usize)
                    .map_err(|e| ImageCompressorError::CompressionError(anyhow::anyhow!(e)))?
                    .clamp(0.0, 1.0);

                let idx = (py * w + px) * 3;
                pixels[idx] = (r * 255.0).round() as u8;
                pixels[idx + 1] = (g * 255.0).round() as u8;
                pixels[idx + 2] = (b * 255.0).round() as u8;
            }
        }

        RgbImage::from_raw(width, height, pixels)
            .ok_or_else(|| ImageCompressorError::InvalidImageSize { width, height })
    }

    /// Compression ratio: original pixels / total TT parameters.
    pub fn compression_ratio(&self) -> f64 {
        let original = (self.original_width as usize) * (self.original_height as usize) * 3;
        let compressed = tt_param_count(&self.tt_r)
            + tt_param_count(&self.tt_g)
            + tt_param_count(&self.tt_b);
        original as f64 / compressed.max(1) as f64
    }
}

fn tt_param_count(tt: &tensor4all_simplett::TensorTrain<f64>) -> usize {
    tt.core_tensors().iter().map(|c| c.len()).sum()
}
```

Add `mod reconstruct;` to `lib.rs`.

**Step 5: Run test to verify it passes**

```bash
cargo nextest run --release -p tensor4all-imagecompressor test_reconstruct_solid_color 2>&1 | tail -10
```
Expected: PASS.

**Step 6: Commit**

```bash
git add crates/tensor4all-imagecompressor/src/
git commit -m "feat(imagecompressor): add reconstruct function"
```

---

### Task 5: Round-trip accuracy test and upscale test

**Files:**
- Modify: `crates/tensor4all-imagecompressor/src/lib.rs` (add tests)

**Step 1: Write the failing tests**

Add to the `#[cfg(test)]` block in `lib.rs`:
```rust
#[test]
fn test_reconstruct_original_size() {
    use image::{DynamicImage, RgbImage, Rgb};
    // 4x4 gradient image
    let mut img_buf = RgbImage::new(4, 4);
    for y in 0..4u32 {
        for x in 0..4u32 {
            img_buf.put_pixel(x, y, Rgb([(x * 60) as u8, (y * 60) as u8, 128u8]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(img_buf.clone());
    let opts = CompressOptions { tolerance: 1e-6, max_rank: None };
    let compressed = compress(&dyn_img, opts).unwrap();
    let reconstructed = compressed.reconstruct(4, 4).unwrap();
    // Each pixel should be within 5/255 of original
    for (orig, recon) in img_buf.pixels().zip(reconstructed.pixels()) {
        for c in 0..3 {
            let diff = (orig[c] as i32 - recon[c] as i32).abs();
            assert!(diff <= 5, "channel {c} diff too large: {diff}");
        }
    }
}

#[test]
fn test_upscale() {
    use image::{DynamicImage, RgbImage, Rgb};
    let mut img_buf = RgbImage::new(4, 4);
    for pixel in img_buf.pixels_mut() {
        *pixel = Rgb([100u8, 150u8, 200u8]);
    }
    let dyn_img = DynamicImage::ImageRgb8(img_buf);
    let opts = CompressOptions { tolerance: 1e-6, max_rank: None };
    let compressed = compress(&dyn_img, opts).unwrap();
    let reconstructed = compressed.reconstruct(8, 8).unwrap();
    assert_eq!(reconstructed.width(), 8);
    assert_eq!(reconstructed.height(), 8);
    // All pixels should be roughly the original color
    for pixel in reconstructed.pixels() {
        assert!((pixel[0] as i32 - 100).abs() <= 5);
        assert!((pixel[1] as i32 - 150).abs() <= 5);
        assert!((pixel[2] as i32 - 200).abs() <= 5);
    }
}

#[test]
fn test_compression_ratio_positive() {
    use image::{DynamicImage, RgbImage, Rgb};
    let mut img_buf = RgbImage::new(4, 4);
    for pixel in img_buf.pixels_mut() {
        *pixel = Rgb([128u8, 64u8, 32u8]);
    }
    let dyn_img = DynamicImage::ImageRgb8(img_buf);
    let opts = CompressOptions::default();
    let compressed = compress(&dyn_img, opts).unwrap();
    assert!(compressed.compression_ratio() > 0.0);
}
```

**Step 2: Run tests to verify they fail**

```bash
cargo nextest run --release -p tensor4all-imagecompressor 2>&1 | tail -20
```
Expected: compile errors (no issues from Task 4 should remain).

**Step 3: Run tests after Task 4 is done**

```bash
cargo nextest run --release -p tensor4all-imagecompressor 2>&1 | tail -20
```
Expected: all tests PASS.

**Step 4: Commit**

```bash
git add crates/tensor4all-imagecompressor/src/lib.rs
git commit -m "test(imagecompressor): add round-trip and upscale tests"
```

---

### Task 6: Lint and final check

**Step 1: Format**

```bash
cargo fmt --all
```

**Step 2: Clippy**

```bash
cargo clippy -p tensor4all-imagecompressor -- -D warnings
```

Fix any warnings.

**Step 3: Full test suite**

```bash
cargo nextest run --release -p tensor4all-imagecompressor 2>&1
```
Expected: all tests PASS.

**Step 4: Commit**

```bash
git add -p
git commit -m "style(imagecompressor): fix clippy warnings and formatting"
```

---

## Notes

- `quanticscrossinterpolate_discrete` requires all grid dimensions equal and power-of-2.
- Grid indices in quanticsgrids are **1-indexed** (`i64`); `TensorTrain::evaluate` uses **0-indexed** `usize`.
- The `Interleaved` unfolding scheme is the default and works well for 2D images.
- For images where `W != H`, both axes use `bits = ceil(log2(max(W,H)))` — some grid cells outside the original image area are set to 0.
- `compression_ratio()` counts total TT parameters across all 3 channels. A value > 1 means compressed.
