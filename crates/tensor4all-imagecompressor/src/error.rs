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
