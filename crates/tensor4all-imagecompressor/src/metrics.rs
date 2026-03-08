use image::RgbImage;
use tensor4all_simplett::AbstractTensorTrain;

use crate::compress::ChannelData;
use crate::CompressedImage;

/// Compute PSNR (Peak Signal-to-Noise Ratio) between two RGB images in dB.
///
/// Returns `f64::INFINITY` when images are identical.
pub fn compute_psnr(original: &RgbImage, reconstructed: &RgbImage) -> f64 {
    assert_eq!(
        original.dimensions(),
        reconstructed.dimensions(),
        "images must have identical dimensions"
    );

    let (width, height) = original.dimensions();
    let n = (width as f64) * (height as f64) * 3.0;
    let mse = original
        .pixels()
        .zip(reconstructed.pixels())
        .map(|(a, b)| {
            (0..3)
                .map(|c| {
                    let d = (a[c] as f64) - (b[c] as f64);
                    d * d
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        / n;

    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }
}

/// Maximum TT bond dimension across RGB channels.
pub fn max_bond_dimension(compressed: &CompressedImage) -> usize {
    fn channel_max_bond(channel: &ChannelData) -> usize {
        match channel {
            ChannelData::Tt { tt, .. } => tt.link_dims().into_iter().max().unwrap_or(1),
            ChannelData::Zero { .. } => 1,
        }
    }

    channel_max_bond(&compressed.ch_r)
        .max(channel_max_bond(&compressed.ch_g))
        .max(channel_max_bond(&compressed.ch_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn test_psnr_identical() {
        let img = RgbImage::from_fn(4, 4, |_, _| Rgb([128u8, 64, 32]));
        let psnr = compute_psnr(&img, &img);
        assert!(
            psnr > 100.0,
            "identical images should have very high PSNR: {psnr}"
        );
    }

    #[test]
    fn test_psnr_different() {
        let a = RgbImage::from_fn(4, 4, |_, _| Rgb([0u8, 0, 0]));
        let b = RgbImage::from_fn(4, 4, |_, _| Rgb([255u8, 255, 255]));
        let psnr = compute_psnr(&a, &b);
        assert!(
            psnr < 1.0,
            "black vs white should have very low PSNR: {psnr}"
        );
    }

    #[test]
    fn test_psnr_small_diff() {
        let a = RgbImage::from_fn(4, 4, |_, _| Rgb([100u8, 100, 100]));
        let b = RgbImage::from_fn(4, 4, |_, _| Rgb([101u8, 100, 100]));
        let psnr = compute_psnr(&a, &b);
        assert!(
            (40.0..60.0).contains(&psnr),
            "small diff PSNR should be around 48-53 dB: {psnr}"
        );
    }
}
