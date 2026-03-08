use image::RgbImage;

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

        let grid_size = 1usize << self.bits;
        let w = width as usize;
        let h = height as usize;
        let source_w = self.original_width as usize;
        let source_h = self.original_height as usize;

        let mut pixels = vec![0u8; w * h * 3];

        for py in 0..h {
            for px in 0..w {
                // Scale output pixel to the original image domain first, then map to QTT grid.
                // This avoids sampling zero-padded tail regions for non-power-of-two inputs.
                let src_x = if w > 1 {
                    ((px as f64) * ((source_w.saturating_sub(1)) as f64) / (w as f64 - 1.0)).round()
                        as usize
                } else {
                    0
                };
                let src_y = if h > 1 {
                    ((py as f64) * ((source_h.saturating_sub(1)) as f64) / (h as f64 - 1.0)).round()
                        as usize
                } else {
                    0
                };
                let gx = src_x.min(source_w.saturating_sub(1)).min(grid_size - 1);
                let gy = src_y.min(source_h.saturating_sub(1)).min(grid_size - 1);

                // QuanticsTensorCI2::evaluate uses 1-indexed grid coordinates.
                let gx_idx = (gx as i64) + 1;
                let gy_idx = (gy as i64) + 1;

                let r = self.ch_r.evaluate(gx_idx, gy_idx)?.clamp(0.0, 1.0);
                let g = self.ch_g.evaluate(gx_idx, gy_idx)?.clamp(0.0, 1.0);
                let b = self.ch_b.evaluate(gx_idx, gy_idx)?.clamp(0.0, 1.0);

                let idx = (py * w + px) * 3;
                pixels[idx] = (r * 255.0).round() as u8;
                pixels[idx + 1] = (g * 255.0).round() as u8;
                pixels[idx + 2] = (b * 255.0).round() as u8;
            }
        }

        RgbImage::from_raw(width, height, pixels)
            .ok_or(ImageCompressorError::InvalidImageSize { width, height })
    }

    /// Compression ratio: original pixel count * 3 channels / total TT parameters.
    ///
    /// Values > 1.0 mean the compressed form is smaller than the original.
    pub fn compression_ratio(&self) -> f64 {
        let original = (self.original_width as usize) * (self.original_height as usize) * 3;
        let compressed =
            self.ch_r.param_count() + self.ch_g.param_count() + self.ch_b.param_count();
        original as f64 / compressed.max(1) as f64
    }
}
