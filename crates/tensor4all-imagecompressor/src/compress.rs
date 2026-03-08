use image::DynamicImage;
use matrixci::util::{ncols, nrows, zeros, Matrix};
use matrixci::{rrlu, RrLUOptions};
use quanticsgrids::{InherentDiscreteGrid, UnfoldingScheme};
use tensor4all_simplett::{tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain};

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

/// One compressed channel stored as a TT in quantics representation.
pub(crate) enum ChannelData {
    /// QTT from direct TT decomposition (non-zero channel).
    Tt {
        tt: TensorTrain<f64>,
        grid: InherentDiscreteGrid,
    },
    /// The channel was identically zero — skip decomposition entirely.
    Zero {
        /// Local dimensions of the TT sites (for compression_ratio).
        local_dims: Vec<usize>,
    },
}

impl ChannelData {
    /// Evaluate the channel at 1-indexed grid coordinates (gx, gy).
    pub(crate) fn evaluate(&self, gx: i64, gy: i64) -> Result<f64, ImageCompressorError> {
        match self {
            ChannelData::Tt { tt, grid } => {
                let quantics = grid.grididx_to_quantics(&[gx, gy]).map_err(|e| {
                    ImageCompressorError::CompressionError(anyhow::anyhow!("{}", e))
                })?;
                // Convert 1-indexed quantics to 0-indexed for TT evaluate
                let quantics_usize: Vec<usize> =
                    quantics.iter().map(|&x| (x - 1) as usize).collect();
                tt.evaluate(&quantics_usize)
                    .map_err(|e| ImageCompressorError::CompressionError(anyhow::anyhow!("{}", e)))
            }
            ChannelData::Zero { .. } => Ok(0.0),
        }
    }

    /// Total number of TT parameters (for compression_ratio).
    pub(crate) fn param_count(&self) -> usize {
        match self {
            ChannelData::Tt { tt, .. } => tt
                .site_tensors()
                .iter()
                .map(|s: &tensor4all_simplett::Tensor3<f64>| s.len())
                .sum(),
            ChannelData::Zero { local_dims } => local_dims.iter().product::<usize>(),
        }
    }
}

/// A compressed image stored as three QTTs (one per RGB channel).
pub struct CompressedImage {
    pub(crate) ch_r: ChannelData,
    pub(crate) ch_g: ChannelData,
    pub(crate) ch_b: ChannelData,
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

/// Build a TT from a full quantics tensor using sequential RRLU factorization.
///
/// The tensor is defined by `values[flat_index]` where the flat index corresponds
/// to the multi-index (s0, s1, ..., s_{N-1}) with local dimensions from the grid.
fn tt_from_quantics_tensor(
    values: &[f64],
    local_dims: &[usize],
    tolerance: f64,
    max_rank: usize,
) -> Result<TensorTrain<f64>, ImageCompressorError> {
    let n = local_dims.len();
    if n == 0 {
        return Err(ImageCompressorError::CompressionError(anyhow::anyhow!(
            "Empty local_dims"
        )));
    }

    // Total number of elements
    let total: usize = local_dims.iter().product();
    assert_eq!(values.len(), total);

    if n == 1 {
        // Single site: TT is just a rank-1 tensor
        let mut t = tensor3_zeros(1, local_dims[0], 1);
        for (s, &v) in values.iter().enumerate().take(local_dims[0]) {
            t.set3(0, s, 0, v);
        }
        return TensorTrain::new(vec![t])
            .map_err(|e| ImageCompressorError::CompressionError(anyhow::anyhow!("{}", e)));
    }

    // Sequential left-to-right factorization (TT-LU decomposition)
    let mut site_tensors: Vec<tensor4all_simplett::Tensor3<f64>> = Vec::with_capacity(n);

    // Current remainder matrix: starts as reshape(values, [d0, d1*d2*...*d_{N-1}])
    let left_dim = local_dims[0];
    let right_dim: usize = local_dims[1..].iter().product();
    let mut remainder: Matrix<f64> = zeros(left_dim, right_dim);
    for i in 0..left_dim {
        for j in 0..right_dim {
            remainder[[i, j]] = values[i * right_dim + j];
        }
    }

    let lu_opts = RrLUOptions {
        max_rank,
        rel_tol: tolerance,
        abs_tol: 0.0,
        left_orthogonal: true,
    };

    // Process sites 0 through N-2
    let mut prev_rank = 1usize;
    for site in 0..n - 1 {
        let d = local_dims[site];
        let nr = nrows(&remainder);

        // remainder has shape (prev_rank * d, remaining_dims)
        assert_eq!(nr, prev_rank * d);

        // RRLU factorize
        let lu = rrlu(&remainder, Some(lu_opts.clone()))
            .map_err(|e| ImageCompressorError::CompressionError(anyhow::anyhow!("{}", e)))?;

        let new_rank = lu.npivots().max(1);
        let left = lu.left(true); // (nr × new_rank)
        let right = lu.right(true); // (new_rank × nc)

        // Build site tensor: shape (prev_rank, d, new_rank)
        let mut t = tensor3_zeros(prev_rank, d, new_rank);
        for l in 0..prev_rank {
            for s in 0..d {
                for r in 0..new_rank {
                    let row = l * d + s;
                    if row < nrows(&left) && r < ncols(&left) {
                        t.set3(l, s, r, left[[row, r]]);
                    }
                }
            }
        }
        site_tensors.push(t);

        if site < n - 2 {
            // Reshape right factor for next iteration
            // right has shape (new_rank, remaining_dims)
            // We need to reshape it as (new_rank * d_{site+1}, remaining_after_next)
            let next_d = local_dims[site + 1];
            let remaining_after_next: usize = local_dims[site + 2..].iter().product();
            assert_eq!(ncols(&right), next_d * remaining_after_next);

            let mut new_remainder: Matrix<f64> = zeros(new_rank * next_d, remaining_after_next);
            for r in 0..new_rank {
                for s in 0..next_d {
                    for j in 0..remaining_after_next {
                        new_remainder[[r * next_d + s, j]] =
                            right[[r, s * remaining_after_next + j]];
                    }
                }
            }
            remainder = new_remainder;
        } else {
            // Last iteration: right factor becomes the last site tensor
            let last_d = local_dims[n - 1];
            assert_eq!(ncols(&right), last_d); // remaining is just d_{N-1} * 1
            let mut t_last = tensor3_zeros(new_rank, last_d, 1);
            for l in 0..new_rank {
                for s in 0..last_d {
                    if l < nrows(&right) && s < ncols(&right) {
                        t_last.set3(l, s, 0, right[[l, s]]);
                    }
                }
            }
            site_tensors.push(t_last);
        }

        prev_rank = new_rank;
    }

    TensorTrain::new(site_tensors)
        .map_err(|e| ImageCompressorError::CompressionError(anyhow::anyhow!("{}", e)))
}

/// Compress an image into a `CompressedImage` using QTT.
pub fn compress(
    img: &DynamicImage,
    opts: CompressOptions,
) -> Result<CompressedImage, ImageCompressorError> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    if w == 0 || h == 0 {
        return Err(ImageCompressorError::InvalidImageSize {
            width: w,
            height: h,
        });
    }

    let bits = bits_for_size(w, h);
    // Build quantics grid
    let rs = vec![bits, bits];
    let grid = InherentDiscreteGrid::builder(&rs)
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .map_err(|e| ImageCompressorError::CompressionError(anyhow::anyhow!("{}", e)))?;

    let local_dims = grid.local_dimensions();
    let total_quantics: usize = local_dims.iter().product();

    let pixels: Vec<u8> = rgb.into_raw();
    let w_usize = w as usize;
    let h_usize = h as usize;

    let max_rank = opts.max_rank.unwrap_or(usize::MAX);

    let compress_channel = |channel: usize| -> Result<ChannelData, ImageCompressorError> {
        // Check if channel is entirely zero
        let has_nonzero = pixels.iter().skip(channel).step_by(3).any(|&v| v != 0);

        if !has_nonzero {
            return Ok(ChannelData::Zero {
                local_dims: local_dims.clone(),
            });
        }

        // Build the full quantics tensor
        let mut values = vec![0.0f64; total_quantics];
        for (flat_idx, value) in values.iter_mut().enumerate() {
            // Convert flat index to multi-index
            let mut multi_idx = vec![0usize; local_dims.len()];
            let mut remainder = flat_idx;
            for i in (0..local_dims.len()).rev() {
                multi_idx[i] = remainder % local_dims[i];
                remainder /= local_dims[i];
            }

            // Convert to 1-indexed quantics for grid
            let quantics_1idx: Vec<i64> = multi_idx.iter().map(|&x| (x as i64) + 1).collect();

            // Convert quantics to grid indices
            let grid_idx = grid
                .quantics_to_grididx(&quantics_1idx)
                .map_err(|e| ImageCompressorError::CompressionError(anyhow::anyhow!("{}", e)))?;

            // grid_idx is 1-indexed
            let gx = (grid_idx[0] - 1) as usize;
            let gy = (grid_idx[1] - 1) as usize;

            if gx < w_usize && gy < h_usize {
                *value = pixels[(gy * w_usize + gx) * 3 + channel] as f64 / 255.0;
            }
            // else: out-of-range pixels are 0.0 (already initialized)
        }

        let tt = tt_from_quantics_tensor(&values, &local_dims, opts.tolerance, max_rank)?;

        Ok(ChannelData::Tt {
            tt,
            grid: grid.clone(),
        })
    };

    let ch_r = compress_channel(0)?;
    let ch_g = compress_channel(1)?;
    let ch_b = compress_channel(2)?;

    Ok(CompressedImage {
        ch_r,
        ch_g,
        ch_b,
        bits,
        original_width: w,
        original_height: h,
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

    #[test]
    fn test_tt_from_quantics_identity() {
        // f(i, j) = i + j on a 4×4 grid, Fused scheme
        let rs = vec![2usize, 2]; // 2 bits per dim → 4×4
        let grid = InherentDiscreteGrid::builder(&rs)
            .with_unfolding_scheme(UnfoldingScheme::Fused)
            .build()
            .unwrap();
        let local_dims = grid.local_dimensions();
        let total: usize = local_dims.iter().product();

        // Build full tensor: f(gx, gy) = gx + gy (1-indexed)
        let mut values = vec![0.0f64; total];
        for flat_idx in 0..total {
            let mut multi_idx = vec![0usize; local_dims.len()];
            let mut remainder = flat_idx;
            for i in (0..local_dims.len()).rev() {
                multi_idx[i] = remainder % local_dims[i];
                remainder /= local_dims[i];
            }
            let quantics_1idx: Vec<i64> = multi_idx.iter().map(|&x| (x as i64) + 1).collect();
            let grid_idx = grid.quantics_to_grididx(&quantics_1idx).unwrap();
            values[flat_idx] = (grid_idx[0] + grid_idx[1]) as f64;
        }

        let tt = tt_from_quantics_tensor(&values, &local_dims, 1e-10, usize::MAX).unwrap();

        // Verify ALL 16 grid points
        let mut max_diff = 0.0f64;
        for gy in 1..=4i64 {
            for gx in 1..=4i64 {
                let expected = (gx + gy) as f64;
                let quantics = grid.grididx_to_quantics(&[gx, gy]).unwrap();
                let quantics_usize: Vec<usize> =
                    quantics.iter().map(|&x| (x - 1) as usize).collect();
                let val = tt.evaluate(&quantics_usize).unwrap();
                let diff = (val - expected).abs();
                max_diff = max_diff.max(diff);
            }
        }
        assert!(max_diff < 1e-8, "TT-LU max error too large: {max_diff}");
    }

    #[test]
    fn test_tt_from_quantics_y_only() {
        // f(x, y) = y (1-indexed) on a 4×4 grid
        let rs = vec![2usize, 2];
        let grid = InherentDiscreteGrid::builder(&rs)
            .with_unfolding_scheme(UnfoldingScheme::Fused)
            .build()
            .unwrap();
        let local_dims = grid.local_dimensions();
        let total: usize = local_dims.iter().product();

        let mut values = vec![0.0f64; total];
        for flat_idx in 0..total {
            let mut multi_idx = vec![0usize; local_dims.len()];
            let mut remainder = flat_idx;
            for i in (0..local_dims.len()).rev() {
                multi_idx[i] = remainder % local_dims[i];
                remainder /= local_dims[i];
            }
            let quantics_1idx: Vec<i64> = multi_idx.iter().map(|&x| (x as i64) + 1).collect();
            let grid_idx = grid.quantics_to_grididx(&quantics_1idx).unwrap();
            values[flat_idx] = grid_idx[1] as f64;
        }

        let tt = tt_from_quantics_tensor(&values, &local_dims, 1e-10, usize::MAX).unwrap();

        let mut max_diff = 0.0f64;
        for gy in 1..=4i64 {
            for gx in 1..=4i64 {
                let expected = gy as f64;
                let quantics = grid.grididx_to_quantics(&[gx, gy]).unwrap();
                let quantics_usize: Vec<usize> =
                    quantics.iter().map(|&x| (x - 1) as usize).collect();
                let val = tt.evaluate(&quantics_usize).unwrap();
                let diff = (val - expected).abs();
                max_diff = max_diff.max(diff);
            }
        }
        assert!(
            max_diff < 1e-8,
            "y-only TT-LU max error too large: {max_diff}"
        );
    }
}
