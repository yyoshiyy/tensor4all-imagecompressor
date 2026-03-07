use image::{Rgb, RgbImage};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Horizontal gradient: black on the left and white on the right.
pub fn horizontal_gradient(width: u32, height: u32) -> RgbImage {
    RgbImage::from_fn(width, height, |x, _| {
        let v = if width > 1 {
            ((x as f64) * 255.0 / ((width - 1) as f64)).round() as u8
        } else {
            128
        };
        Rgb([v, v, v])
    })
}

/// Diagonal gradient: intensity proportional to x + y.
pub fn diagonal_gradient(width: u32, height: u32) -> RgbImage {
    let max_sum = (width.saturating_sub(1) + height.saturating_sub(1)) as f64;
    RgbImage::from_fn(width, height, |x, y| {
        let v = if max_sum > 0.0 {
            (((x + y) as f64) * 255.0 / max_sum).round() as u8
        } else {
            128
        };
        Rgb([v, v, v])
    })
}

/// Checkerboard with configurable square size.
pub fn checkerboard(width: u32, height: u32, square_size: u32) -> RgbImage {
    let square = square_size.max(1);
    RgbImage::from_fn(width, height, |x, y| {
        let cx = x / square;
        let cy = y / square;
        let v = if (cx + cy).is_multiple_of(2) { 255 } else { 0 };
        Rgb([v, v, v])
    })
}

/// Concentric circle-like pattern using radial sinusoid from image center.
pub fn concentric_circles(width: u32, height: u32) -> RgbImage {
    let cx = (width as f64) * 0.5;
    let cy = (height as f64) * 0.5;
    let max_r = (cx * cx + cy * cy).sqrt();

    RgbImage::from_fn(width, height, |x, y| {
        let dx = (x as f64) - cx;
        let dy = (y as f64) - cy;
        let r = (dx * dx + dy * dy).sqrt();
        let v = if max_r > 0.0 {
            ((r / max_r * 8.0 * std::f64::consts::PI).sin() * 127.5 + 127.5)
                .round()
                .clamp(0.0, 255.0) as u8
        } else {
            128
        };
        Rgb([v, v, v])
    })
}

/// Random noise pattern with deterministic seed.
pub fn random_noise(width: u32, height: u32, seed: u64) -> RgbImage {
    let mut rng = StdRng::seed_from_u64(seed);
    RgbImage::from_fn(width, height, |_x, _y| {
        Rgb([rng.random::<u8>(), rng.random::<u8>(), rng.random::<u8>()])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horizontal_gradient_corners() {
        let img = horizontal_gradient(4, 4);
        assert_eq!(img.get_pixel(0, 0)[0], 0);
        assert_eq!(img.get_pixel(3, 0)[0], 255);
    }

    #[test]
    fn test_checkerboard_pattern() {
        let img = checkerboard(8, 8, 2);
        assert_eq!(img.get_pixel(0, 0)[0], 255);
        assert_eq!(img.get_pixel(2, 0)[0], 0);
    }

    #[test]
    fn test_image_dimensions() {
        assert_eq!(horizontal_gradient(16, 32).dimensions(), (16, 32));
        assert_eq!(checkerboard(64, 64, 8).dimensions(), (64, 64));
        assert_eq!(concentric_circles(32, 32).dimensions(), (32, 32));
        assert_eq!(diagonal_gradient(128, 128).dimensions(), (128, 128));
        assert_eq!(random_noise(64, 64, 42).dimensions(), (64, 64));
    }
}
