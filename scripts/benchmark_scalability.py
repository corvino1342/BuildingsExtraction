"""
benchmark_scalability.py
-------------------------
Empirical scalability benchmark for the raster building-extraction pipeline
(predict.py's `raster` subcommand / RasterWindowDataset).

For a given raster and trained model, this script sweeps a list of batch
sizes and, for each one, measures:

  * wall-clock inference time over the whole raster
  * throughput in megapixels/second (Mpix/s)
  * throughput in km^2/min (using the raster's own pixel resolution, or an
    override via --pixel_size_m)
  * peak GPU memory (if running on CUDA)

The output is a CSV table plus a small PNG plot (throughput and memory vs.
batch size). Both are meant to be dropped straight into the "Scalability"
subsection of the paper as empirical evidence, instead of a prose-only claim.

USAGE
-----
Place this file next to predict.py (e.g. in scripts/) and run, from the
repo root, something like:

    python -m scripts.benchmark_scalability \
        --input /data/orthophoto.tif \
        --model_path runs/WHUBuildingDataset/unetLL_bce_.../best_model.pth \
        --arch unetLL --in_channels 3 \
        --tile_size_px 256 --overlap 32 \
        --batch_sizes 1 4 8 16 32 64 \
        --device cuda \
        --output_dir experiments/scalability

Notes
-----
- No output masks/probability maps are written by this script -- it only
  times the forward pass over the whole raster, exactly as `predict.py
  raster` would do it (same dataset class, same normalization, same
  windowing), so the numbers are directly representative of the real
  pipeline's cost.
- If your raster is not georeferenced (no CRS/transform), pass
  --pixel_size_m explicitly, or the km^2/min column will be left empty.
- If a batch size is too large for available memory, the script prints the
  CUDA OOM and continues with the next size, recording it as failed.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# These come straight from predict.py -- reusing the exact same dataset /
# inference code path used in production (`predict.py raster`) so the
# benchmark numbers are representative of the real pipeline.
from predict import (
    build_model,          # noqa: F401 (kept for parity with predict.py's public API)
    resolve_device,
    load_model,
    predict_batch,
    RasterWindowDataset,
)


# --------------------------------------------------
# CLI
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep batch size on the raster inference pipeline and "
                    "report throughput / peak GPU memory."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a (georeferenced) raster, e.g. a GeoTIFF.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--arch", type=str, choices=["unet", "unetL", "unetLL"], default="unetLL")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--tile_size_px", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--normalize", type=str, choices=["auto", "divide255", "none"], default="auto")

    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                        help="List of batch sizes to sweep, e.g. --batch_sizes 1 4 8 16 32")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"])
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeat each batch size N times and report the median "
                            "(reduces noise from OS/disk jitter).")
    parser.add_argument("--warmup_batches", type=int, default=1,
                        help="Number of batches to run once before timing starts, "
                            "to exclude one-off CUDA/cuDNN warm-up cost from the measurement.")
    parser.add_argument("--pixel_size_m", type=float, default=None,
                        help="Override ground sample distance in meters/pixel "
                            "(use this if the raster has no CRS/transform).")
    parser.add_argument("--output_dir", type=str, default="experiments/scalability")

    return parser.parse_args()


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def get_pixel_area_km2(raster_path, pixel_size_m=None):
    """Ground area (km^2) covered by one pixel, from the raster's own
    affine transform unless overridden."""
    if pixel_size_m is not None:
        return (pixel_size_m ** 2) / 1e6

    import rasterio
    with rasterio.open(raster_path) as src:
        transform = src.transform
        px_x = abs(transform.a)
        px_y = abs(transform.e)

    if px_x == 0 or px_y == 0 or px_x > 1e4:
        # Not georeferenced (identity transform, pixel units), or clearly
        # not meters -- caller should pass --pixel_size_m instead.
        return None
    return (px_x * px_y) / 1e6


def run_one_pass(model, device, dataset, batch_size, num_workers, warmup_batches):
    """Times a full sweep over `dataset` at the given batch size. Returns
    (elapsed_seconds, n_tile_pixels, peak_mem_MB_or_None)."""

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Warm-up (not timed): first CUDA kernels / cuDNN autotune are much
    # slower than steady-state, and would otherwise bias small-raster runs.
    if warmup_batches > 0:
        done = 0
        for idxs, tensors in loader:
            predict_batch(model, tensors, device)
            done += 1
            if done >= warmup_batches:
                break

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_pixels = 0
    for idxs, tensors in loader:
        probs = predict_batch(model, tensors, device)
        n_pixels += int(probs.shape[0]) * int(probs.shape[1]) * int(probs.shape[2])

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mem_mb = None
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return elapsed, n_pixels, peak_mem_mb


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    device = resolve_device(args.device)

    print(f"Loading model ({args.arch}, {args.in_channels} ch) from {args.model_path} on {device}...")
    model, device = load_model(args.model_path, args.arch, args.in_channels, device)

    px_area_km2 = get_pixel_area_km2(args.input, args.pixel_size_m)
    if px_area_km2 is None:
        print("WARNING: could not determine ground pixel size (no CRS/transform found). "
            "km^2/min column will be left empty -- pass --pixel_size_m to enable it.")

    results = []

    for bs in args.batch_sizes:
        # Rebuilding the dataset per batch size is cheap: RasterWindowDataset
        # only opens the raster to read its header/shape in __init__.
        dataset = RasterWindowDataset(
            args.input, args.tile_size_px, args.overlap, args.in_channels, args.normalize
        )

        elapsed_runs, pixels_runs = [], []
        peak_mem_mb = None
        failed = False

        for r in range(args.repeats):
            try:
                elapsed, n_pixels, mem = run_one_pass(
                    model, device, dataset, bs, args.num_workers, args.warmup_batches
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"batch_size={bs}: CUDA OOM, skipping. ({e})")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    failed = True
                    break
                raise
            elapsed_runs.append(elapsed)
            pixels_runs.append(n_pixels)
            peak_mem_mb = mem if mem is not None else peak_mem_mb

        if failed:
            results.append({
                "batch_size": bs, "elapsed_s": None, "n_tile_pixels": None,
                "mpixels_per_s": None, "km2_per_min": None,
                "peak_mem_MB": None, "status": "OOM",
            })
            continue

        elapsed = float(np.median(elapsed_runs))
        n_pixels = int(np.median(pixels_runs))
        mpixels_per_s = (n_pixels / 1e6) / elapsed
        km2_per_min = (n_pixels * px_area_km2 / elapsed * 60) if px_area_km2 else None

        row = {
            "batch_size": bs,
            "elapsed_s": round(elapsed, 3),
            "n_tile_pixels": n_pixels,
            "mpixels_per_s": round(mpixels_per_s, 3),
            "km2_per_min": round(km2_per_min, 4) if km2_per_min is not None else None,
            "peak_mem_MB": round(peak_mem_mb, 1) if peak_mem_mb is not None else None,
            "status": "ok",
        }
        results.append(row)

        msg = f"batch_size={bs:>4}  time={elapsed:7.2f}s  throughput={mpixels_per_s:8.2f} MPix/s"
        if km2_per_min is not None:
            msg += f"  ({km2_per_min:7.3f} km^2/min)"
        if peak_mem_mb is not None:
            msg += f"  peak_mem={peak_mem_mb:8.1f} MB"
        print(msg)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "scalability_results.csv"
    fieldnames = ["batch_size", "elapsed_s", "n_tile_pixels", "mpixels_per_s",
                "km2_per_min", "peak_mem_MB", "status"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results table -> {csv_path}")

    # --------------------------------------------------
    # Plot (throughput + memory vs batch size)
    # --------------------------------------------------
    ok_rows = [r for r in results if r["status"] == "ok"]
    if ok_rows:
        try:
            import matplotlib.pyplot as plt

            bss = [r["batch_size"] for r in ok_rows]
            thr = [r["mpixels_per_s"] for r in ok_rows]
            mem = [r["peak_mem_MB"] for r in ok_rows]

            fig, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(bss, thr, "o-", color="tab:blue", label="Throughput (MPix/s)")
            ax1.set_xlabel("Batch size")
            ax1.set_ylabel("Throughput (MPix/s)", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.set_xscale("log", base=2)

            if device.type == "cuda" and all(m is not None for m in mem):
                ax2 = ax1.twinx()
                ax2.plot(bss, mem, "s--", color="tab:red", label="Peak GPU mem (MB)")
                ax2.set_ylabel("Peak GPU memory (MB)", color="tab:red")
                ax2.tick_params(axis="y", labelcolor="tab:red")

            fig.suptitle(f"Inference scalability vs. batch size ({args.arch}, "
                        f"tile {args.tile_size_px}px, overlap {args.overlap}px)")
            fig.tight_layout()
            plot_path = out_dir / "scalability_plot.png"
            fig.savefig(plot_path, dpi=150)
            print(f"Saved plot -> {plot_path}")
        except ImportError:
            print("matplotlib not available, skipping plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
