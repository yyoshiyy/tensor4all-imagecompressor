//! Image compression using Quantics Tensor Train (QTT).
//!
//! Compresses PNG/JPEG images into QTT format and reconstructs
//! them at arbitrary resolution.

mod compress;
mod error;
mod reconstruct;

pub use compress::{compress, CompressOptions, CompressedImage};
pub use error::ImageCompressorError;

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, Rgb, RgbImage};

    #[test]
    fn test_error_display() {
        let e = ImageCompressorError::InvalidImageSize {
            width: 0,
            height: 0,
        };
        assert!(e.to_string().contains("0"));
    }

    #[test]
    fn test_reconstruct_solid_color() {
        let mut img_buf = RgbImage::new(4, 4);
        for pixel in img_buf.pixels_mut() {
            *pixel = Rgb([255u8, 0, 0]);
        }
        let dyn_img = DynamicImage::ImageRgb8(img_buf);
        let opts = CompressOptions {
            tolerance: 1e-6,
            max_rank: None,
        };
        let compressed = compress(&dyn_img, opts).unwrap();
        let reconstructed = compressed.reconstruct(4, 4).unwrap();
        for pixel in reconstructed.pixels() {
            assert!(pixel[0] > 200, "red channel should be high");
            assert!(pixel[1] < 10, "green channel should be near zero");
            assert!(pixel[2] < 10, "blue channel should be near zero");
        }
    }

    #[test]
    fn test_reconstruct_original_size() {
        let mut img_buf = RgbImage::new(4, 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                img_buf.put_pixel(x, y, Rgb([(x * 60) as u8, (y * 60) as u8, 128u8]));
            }
        }
        let dyn_img = DynamicImage::ImageRgb8(img_buf.clone());
        let opts = CompressOptions {
            tolerance: 1e-6,
            max_rank: None,
        };
        let compressed = compress(&dyn_img, opts).unwrap();
        let reconstructed = compressed.reconstruct(4, 4).unwrap();
        for (orig, recon) in img_buf.pixels().zip(reconstructed.pixels()) {
            for c in 0..3 {
                let diff = (orig[c] as i32 - recon[c] as i32).abs();
                assert!(diff <= 5, "channel {c} diff too large: {diff}");
            }
        }
    }

    #[test]
    fn test_upscale() {
        let mut img_buf = RgbImage::new(4, 4);
        for pixel in img_buf.pixels_mut() {
            *pixel = Rgb([100u8, 150u8, 200u8]);
        }
        let dyn_img = DynamicImage::ImageRgb8(img_buf);
        let opts = CompressOptions {
            tolerance: 1e-6,
            max_rank: None,
        };
        let compressed = compress(&dyn_img, opts).unwrap();
        let reconstructed = compressed.reconstruct(8, 8).unwrap();
        assert_eq!(reconstructed.width(), 8);
        assert_eq!(reconstructed.height(), 8);
        for pixel in reconstructed.pixels() {
            assert!((pixel[0] as i32 - 100).abs() <= 5);
            assert!((pixel[1] as i32 - 150).abs() <= 5);
            assert!((pixel[2] as i32 - 200).abs() <= 5);
        }
    }

    #[test]
    fn test_debug_gradient_pixels() {
        let mut img_buf = RgbImage::new(4, 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                img_buf.put_pixel(x, y, Rgb([(x * 60) as u8, (y * 60) as u8, 128u8]));
            }
        }
        let dyn_img = DynamicImage::ImageRgb8(img_buf.clone());
        let opts = CompressOptions {
            tolerance: 1e-6,
            max_rank: None,
        };
        let compressed = compress(&dyn_img, opts).unwrap();
        let reconstructed = compressed.reconstruct(4, 4).unwrap();
        for (i, (orig, recon)) in img_buf.pixels().zip(reconstructed.pixels()).enumerate() {
            let px = i % 4;
            let py = i / 4;
            for c in 0..3 {
                let diff = (orig[c] as i32 - recon[c] as i32).abs();
                if diff > 5 {
                    eprintln!(
                        "pixel ({px},{py}) ch{c}: orig={} recon={} diff={diff}",
                        orig[c], recon[c]
                    );
                }
            }
        }
    }

    #[test]
    fn test_compression_ratio_positive() {
        let mut img_buf = RgbImage::new(4, 4);
        for pixel in img_buf.pixels_mut() {
            *pixel = Rgb([128u8, 64u8, 32u8]);
        }
        let dyn_img = DynamicImage::ImageRgb8(img_buf);
        let opts = CompressOptions::default();
        let compressed = compress(&dyn_img, opts).unwrap();
        assert!(compressed.compression_ratio() > 0.0);
    }
}
