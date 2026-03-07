//! QTT image compression experiment runner.
//!
//! Usage:
//!   cargo run --release -p tensor4all-imagecompressor --example image_experiment -- \
//!     phase1 --input docs/experiments/images/test.png
//!
//!   cargo run --release -p tensor4all-imagecompressor --example image_experiment -- \
//!     phase2 [--size 512] [--max-rank 128]
//!
//!   cargo run --release -p tensor4all-imagecompressor --example image_experiment -- \
//!     phase3 --input-dir docs/experiments/images/ [--max-rank 128]
//!
//!   cargo run --release -p tensor4all-imagecompressor --example image_experiment -- report

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use image::{DynamicImage, RgbImage};
use tensor4all_imagecompressor::{compress, metrics, synthetic, CompressOptions};

const RESULTS_DIR: &str = "docs/experiments/results";

#[derive(Debug, Clone)]
struct ExperimentResult {
    name: String,
    tolerance: f64,
    compress_time_secs: f64,
    reconstruct_time_secs: f64,
    compression_ratio: f64,
    psnr_db: f64,
    max_bond_dim: usize,
    original_size: (u32, u32),
}

impl ExperimentResult {
    fn markdown_table_header() -> &'static str {
        "| Name | Tolerance | Compress (s) | Reconstruct (s) | Ratio | PSNR (dB) | Max Bond Dim | Size |\n\
         |------|-----------|-------------|----------------|-------|-----------|-------------|------|"
    }

    fn markdown_table_row(&self) -> String {
        format!(
            "| {} | {:.0e} | {:.2} | {:.2} | {:.2} | {:.1} | {} | {}x{} |",
            self.name,
            self.tolerance,
            self.compress_time_secs,
            self.reconstruct_time_secs,
            self.compression_ratio,
            self.psnr_db,
            self.max_bond_dim,
            self.original_size.0,
            self.original_size.1
        )
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        bail!("missing phase");
    }

    std::fs::create_dir_all(RESULTS_DIR).context("failed to create results directory")?;

    match args[1].as_str() {
        "phase1" => {
            let input = argument_value(&args, "--input")
                .ok_or_else(|| anyhow::anyhow!("--input <path> is required for phase1"))?;
            run_phase1(Path::new(&input))
        }
        "phase2" => {
            let size = parse_optional_u32_flag(&args, "--size")?.unwrap_or(512);
            let max_rank = parse_optional_usize_flag(&args, "--max-rank")?;
            run_phase2(size, max_rank)
        }
        "phase3" => {
            let input_dir = argument_value(&args, "--input-dir")
                .ok_or_else(|| anyhow::anyhow!("--input-dir <path> is required for phase3"))?;
            let max_rank = parse_optional_usize_flag(&args, "--max-rank")?;
            run_phase3(Path::new(&input_dir), max_rank)
        }
        "report" => generate_combined_report(),
        _ => {
            print_usage();
            bail!("unknown phase: {}", args[1]);
        }
    }
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  image_experiment phase1 --input <image>");
    eprintln!("  image_experiment phase2 [--size <px>] [--max-rank <rank>]");
    eprintln!("  image_experiment phase3 --input-dir <dir> [--max-rank <rank>]");
    eprintln!("  image_experiment report");
}

fn argument_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| {
        if i + 1 < args.len() {
            Some(args[i + 1].clone())
        } else {
            None
        }
    })
}

fn parse_optional_u32_flag(args: &[String], flag: &str) -> Result<Option<u32>> {
    match argument_value(args, flag) {
        Some(value) => {
            Ok(Some(value.parse::<u32>().with_context(|| {
                format!("failed to parse {flag} as u32")
            })?))
        }
        None => Ok(None),
    }
}

fn parse_optional_usize_flag(args: &[String], flag: &str) -> Result<Option<usize>> {
    match argument_value(args, flag) {
        Some(value) => {
            Ok(Some(value.parse::<usize>().with_context(|| {
                format!("failed to parse {flag} as usize")
            })?))
        }
        None => Ok(None),
    }
}

fn tol_tag(tol: f64) -> String {
    format!("{tol:.0e}")
        .replace('+', "")
        .replace('.', "")
        .replace('-', "m")
}

fn run_experiment(
    name: &str,
    image: &DynamicImage,
    tolerance: f64,
    max_rank: Option<usize>,
    output: Option<&Path>,
) -> Result<ExperimentResult> {
    let original = image.to_rgb8();
    let (width, height) = original.dimensions();
    println!("  [{name}] tol={tolerance:.0e} size={width}x{height}");

    let options = CompressOptions {
        tolerance,
        max_rank,
    };

    let t0 = Instant::now();
    let compressed = compress(image, options).context("compression failed")?;
    let compress_time = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let reconstructed = compressed
        .reconstruct(width, height)
        .context("reconstruction failed")?;
    let reconstruct_time = t1.elapsed().as_secs_f64();

    if let Some(path) = output {
        reconstructed
            .save(path)
            .with_context(|| format!("failed to save {}", path.display()))?;
    }

    let psnr_db = metrics::compute_psnr(&original, &reconstructed);
    let compression_ratio = compressed.compression_ratio();
    let max_bond_dim = metrics::max_bond_dimension(&compressed);

    println!(
        "    compress={compress_time:.2}s reconstruct={reconstruct_time:.2}s ratio={compression_ratio:.2} PSNR={psnr_db:.2}dB bond={max_bond_dim}"
    );

    Ok(ExperimentResult {
        name: name.to_string(),
        tolerance,
        compress_time_secs: compress_time,
        reconstruct_time_secs: reconstruct_time,
        compression_ratio,
        psnr_db,
        max_bond_dim,
        original_size: (width, height),
    })
}

fn run_phase1(input: &Path) -> Result<()> {
    println!("=== Phase 1: Feasibility Check ===");
    let image =
        image::open(input).with_context(|| format!("failed to open {}", input.display()))?;
    let name = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("input");

    let original_path = PathBuf::from(RESULTS_DIR).join(format!("phase1_{name}_original.png"));
    let reconstructed_path =
        PathBuf::from(RESULTS_DIR).join(format!("phase1_{name}_reconstructed.png"));

    image
        .to_rgb8()
        .save(&original_path)
        .with_context(|| format!("failed to save {}", original_path.display()))?;
    let result = run_experiment(name, &image, 1e-3, None, Some(&reconstructed_path))?;

    let report = format!(
        "# Phase 1: Feasibility Check\n\n\
         Input: `{}`\n\n\
         {}\n{}\n\n\
         | Item | Image |\n|---|---|\n\
         | Original | ![original]({}) |\n\
         | Reconstructed | ![reconstructed]({}) |\n",
        input.display(),
        ExperimentResult::markdown_table_header(),
        result.markdown_table_row(),
        original_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy(),
        reconstructed_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    );

    let report_path = PathBuf::from(RESULTS_DIR).join("phase1_report.md");
    std::fs::write(&report_path, report)
        .with_context(|| format!("failed to write {}", report_path.display()))?;
    println!("saved {}", report_path.display());
    Ok(())
}

fn run_phase2(size: u32, max_rank: Option<usize>) -> Result<()> {
    println!("=== Phase 2: Synthetic Images ===");
    let tolerances = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6];

    let patterns: Vec<(&str, RgbImage)> = vec![
        (
            "horizontal_gradient",
            synthetic::horizontal_gradient(size, size),
        ),
        (
            "diagonal_gradient",
            synthetic::diagonal_gradient(size, size),
        ),
        ("checkerboard", synthetic::checkerboard(size, size, 32)),
        (
            "concentric_circles",
            synthetic::concentric_circles(size, size),
        ),
        ("random_noise", synthetic::random_noise(size, size, 42)),
    ];

    let mut all_results = Vec::new();
    for (pattern_name, image) in &patterns {
        println!("\nPattern: {pattern_name}");
        let original_path =
            PathBuf::from(RESULTS_DIR).join(format!("phase2_{pattern_name}_original.png"));
        image
            .save(&original_path)
            .with_context(|| format!("failed to save {}", original_path.display()))?;

        let dynamic = DynamicImage::ImageRgb8(image.clone());
        for &tol in &tolerances {
            let recon_path = PathBuf::from(RESULTS_DIR)
                .join(format!("phase2_{pattern_name}_tol{}.png", tol_tag(tol)));
            let result = run_experiment(pattern_name, &dynamic, tol, max_rank, Some(&recon_path))?;
            all_results.push(result);
        }
    }

    let mut report = String::new();
    report.push_str("# Phase 2: Synthetic Images\n\n");
    report.push_str(&format!("Image size: {size}x{size}\n\n"));
    report.push_str("## Results\n\n");
    report.push_str(ExperimentResult::markdown_table_header());
    report.push('\n');
    for result in &all_results {
        report.push_str(&result.markdown_table_row());
        report.push('\n');
    }

    report.push_str("\n## Visual Comparison\n\n");
    report.push_str("Rows show reconstructions for each tolerance.\n\n");
    for (pattern_name, _) in &patterns {
        report.push_str(&format!("### {pattern_name}\n\n"));
        report.push_str(&format!(
            "Original: ![original](phase2_{pattern_name}_original.png)\n\n"
        ));
        report.push_str("| Tolerance | Reconstruction |\n|-----------|----------------|\n");
        for &tol in &tolerances {
            let filename = format!("phase2_{pattern_name}_tol{}.png", tol_tag(tol));
            report.push_str(&format!("| {tol:.0e} | ![{filename}]({filename}) |\n"));
        }
        report.push('\n');
    }

    let report_path = PathBuf::from(RESULTS_DIR).join("phase2_report.md");
    std::fs::write(&report_path, report)
        .with_context(|| format!("failed to write {}", report_path.display()))?;
    println!("saved {}", report_path.display());
    Ok(())
}

fn run_phase3(input_dir: &Path, max_rank: Option<usize>) -> Result<()> {
    println!("=== Phase 3: Natural Images ===");
    let mut image_paths = collect_input_images(input_dir)?;
    if image_paths.is_empty() {
        bail!("no input images found in {}", input_dir.display());
    }
    image_paths.sort();
    println!("Found {} input images", image_paths.len());

    let tolerances = [1e-1, 1e-2, 1e-3, 1e-4];
    let mut report = String::new();
    report.push_str("# Phase 3: Natural Images\n\n");
    report.push_str("## 1. Tolerance vs Quality/Compression\n\n");
    report.push_str(ExperimentResult::markdown_table_header());
    report.push('\n');

    for path in &image_paths {
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
        println!("\nImage: {name}");
        let image =
            image::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let original_path = PathBuf::from(RESULTS_DIR).join(format!("phase3_{name}_original.png"));
        image
            .to_rgb8()
            .save(&original_path)
            .with_context(|| format!("failed to save {}", original_path.display()))?;

        for &tol in &tolerances {
            let recon_path =
                PathBuf::from(RESULTS_DIR).join(format!("phase3_{name}_tol{}.png", tol_tag(tol)));
            let result = run_experiment(name, &image, tol, max_rank, Some(&recon_path))?;
            report.push_str(&result.markdown_table_row());
            report.push('\n');
        }
    }

    report.push_str("\n## 2. Super-Resolution Demo\n\n");
    let demo_path = &image_paths[0];
    let demo_name = demo_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("demo");
    let demo_image = image::open(demo_path)
        .with_context(|| format!("failed to open {}", demo_path.display()))?;
    let demo_original = demo_image.to_rgb8();
    let (orig_w, orig_h) = demo_original.dimensions();
    let up_w = orig_w * 2;
    let up_h = orig_h * 2;

    let compressed = compress(
        &demo_image,
        CompressOptions {
            tolerance: 1e-3,
            max_rank,
        },
    )
    .context("compression failed in super-resolution demo")?;
    let t0 = Instant::now();
    let qtt_upscaled = compressed
        .reconstruct(up_w, up_h)
        .context("QTT upscaling failed")?;
    let qtt_time = t0.elapsed().as_secs_f64();
    let qtt_upscaled_path =
        PathBuf::from(RESULTS_DIR).join(format!("phase3_{demo_name}_upscaled_qtt.png"));
    qtt_upscaled
        .save(&qtt_upscaled_path)
        .with_context(|| format!("failed to save {}", qtt_upscaled_path.display()))?;

    let nn_upscaled = image::imageops::resize(
        &demo_original,
        up_w,
        up_h,
        image::imageops::FilterType::Nearest,
    );
    let nn_upscaled_path =
        PathBuf::from(RESULTS_DIR).join(format!("phase3_{demo_name}_upscaled_nn.png"));
    nn_upscaled
        .save(&nn_upscaled_path)
        .with_context(|| format!("failed to save {}", nn_upscaled_path.display()))?;

    report.push_str(&format!(
        "Demo image: `{demo_name}` ({orig_w}x{orig_h} -> {up_w}x{up_h})\n\n\
         QTT upscale time: {qtt_time:.2} s\n\n\
         | Method | Image |\n|--------|-------|\n\
         | Original | ![original](phase3_{demo_name}_original.png) |\n\
         | QTT upscale | ![qtt](phase3_{demo_name}_upscaled_qtt.png) |\n\
         | Nearest-neighbor | ![nn](phase3_{demo_name}_upscaled_nn.png) |\n\n"
    ));

    report.push_str("## 3. JPEG Comparison\n\n");
    report.push_str("QTT (`tol=1e-2`) and JPEG are compared at similar compressed byte sizes.\n\n");
    report.push_str("| Image | QTT Ratio | QTT PSNR (dB) | JPEG Quality | JPEG PSNR (dB) |\n");
    report.push_str("|-------|-----------|---------------|--------------|----------------|\n");

    for path in &image_paths {
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
        let image =
            image::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let original = image.to_rgb8();
        let (w, h) = original.dimensions();
        let original_bytes = (w as usize) * (h as usize) * 3;

        let compressed = compress(
            &image,
            CompressOptions {
                tolerance: 1e-2,
                max_rank,
            },
        )
        .context("compression failed in JPEG comparison")?;
        let qtt_ratio = compressed.compression_ratio();
        let qtt_reconstructed = compressed
            .reconstruct(w, h)
            .context("reconstruction failed in JPEG comparison")?;
        let qtt_psnr = metrics::compute_psnr(&original, &qtt_reconstructed);

        let target_bytes = ((original_bytes as f64) / qtt_ratio).max(1.0).round() as usize;
        let (jpeg_quality, jpeg_psnr) = find_jpeg_match(&original, target_bytes)?;

        report.push_str(&format!(
            "| {name} | {qtt_ratio:.2} | {qtt_psnr:.1} | {jpeg_quality} | {jpeg_psnr:.1} |\n"
        ));
    }

    let report_path = PathBuf::from(RESULTS_DIR).join("phase3_report.md");
    std::fs::write(&report_path, report)
        .with_context(|| format!("failed to write {}", report_path.display()))?;
    println!("saved {}", report_path.display());
    Ok(())
}

fn collect_input_images(input_dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = std::fs::read_dir(input_dir)
        .with_context(|| format!("failed to read {}", input_dir.display()))?;
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry?.path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        let ext = ext.to_ascii_lowercase();
        if ext == "png" || ext == "jpg" || ext == "jpeg" {
            paths.push(path);
        }
    }
    Ok(paths)
}

fn find_jpeg_match(original: &RgbImage, target_bytes: usize) -> Result<(u8, f64)> {
    use image::ImageEncoder;
    use std::io::Cursor;

    let (w, h) = original.dimensions();
    let mut best_quality = 50u8;
    let mut best_diff = usize::MAX;

    for quality in (5..=95).step_by(5) {
        let mut buffer = Cursor::new(Vec::new());
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, quality).write_image(
            original.as_raw(),
            w,
            h,
            image::ExtendedColorType::Rgb8,
        )?;
        let jpeg_size = buffer.into_inner().len();
        let diff = jpeg_size.abs_diff(target_bytes);
        if diff < best_diff {
            best_diff = diff;
            best_quality = quality;
        }
    }

    let mut buffer = Cursor::new(Vec::new());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, best_quality).write_image(
        original.as_raw(),
        w,
        h,
        image::ExtendedColorType::Rgb8,
    )?;
    let jpeg_bytes = buffer.into_inner();
    let decoded = image::load_from_memory_with_format(&jpeg_bytes, image::ImageFormat::Jpeg)?;
    let jpeg_rgb = decoded.to_rgb8();
    let psnr = metrics::compute_psnr(original, &jpeg_rgb);

    Ok((best_quality, psnr))
}

fn generate_combined_report() -> Result<()> {
    let results_dir = PathBuf::from(RESULTS_DIR);
    let mut report = String::from("# QTT Image Compression Experiment Report\n\n");
    report.push_str("Generated by `image_experiment`.\n\n---\n\n");

    for (phase, filename) in [
        ("Phase 1", "phase1_report.md"),
        ("Phase 2", "phase2_report.md"),
        ("Phase 3", "phase3_report.md"),
    ] {
        let path = results_dir.join(filename);
        if path.exists() {
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            report.push_str(&content);
            report.push_str("\n\n---\n\n");
        } else {
            report.push_str(&format!("## {phase}\n\n_Not run yet._\n\n---\n\n"));
        }
    }

    let output = results_dir.join("report.md");
    std::fs::write(&output, report)
        .with_context(|| format!("failed to write {}", output.display()))?;
    println!("saved {}", output.display());
    Ok(())
}
