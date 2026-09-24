"""
benchmark_parallel.py
----------------------
Empirical multi-process scalability benchmark for the raster inference
pipeline: splits the raster's tile windows across N worker processes and
measures wall-clock speedup / parallel efficiency relative to a
single-process baseline that processes the same tiles sequentially.

This is the "distributed tiling" precedent used in large-scale
building-extraction mapping (e.g. Yang et al., 2018, CNN-based CONUS-scale
mapping): independent tiles are embarrassingly parallel, so splitting them
across workers (processes / GPUs / machines) should yield close-to-linear
speedup, bounded by I/O and per-device memory. This script produces the
numbers to support that claim with your own hardware and data, rather than
citing the precedent alone.

USAGE
-----
Single GPU, testing process-level parallelism (workers will contend for the
same device -- useful to show the pipeline degrades gracefully rather than
crashing, and to find the point of diminishing returns):

    python -m scripts.benchmark_parallel \
        --input /data/orthophoto.tif \
        --model_path runs/.../best_model.pth --arch unetLL --in_channels 3 \
        --tile_size_px 256 --overlap 32 --batch_size 16 \
        --worker_counts 1 2 4 \
        --device cuda \
        --output_dir experiments/scalability

Multiple GPUs (one worker per device -- the realistic "wider system" setup):

    python -m scripts.benchmark_parallel \
        --input /data/orthophoto.tif \
        --model_path runs/.../best_model.pth --arch unetLL --in_channels 3 \
        --tile_size_px 256 --overlap 32 --batch_size 16 \
        --worker_counts 1 2 4 \
        --devices cuda:0,cuda:1,cuda:2,cuda:3 \
        --output_dir experiments/scalability

CPU-only machine (e.g. to demonstrate the approach without a GPU cluster):

    python -m scripts.benchmark_parallel ... --device cpu --worker_counts 1 2 4 8

Output
------
A JSON file with per-worker timings plus a CSV summary
(n_workers, wall_time_s, speedup_vs_1worker, parallel_efficiency) that maps
directly onto a "scalability" table/plot for the paper.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

from predict import (
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
        description="Split raster tiles across N worker processes and measure "
                    "speedup vs. a single-process baseline."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--arch", type=str, choices=["unet", "unetL", "unetLL"], default="unetLL")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--tile_size_px", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--normalize", type=str, choices=["auto", "divide255", "none"], default="auto")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size used *within* each worker.")

    parser.add_argument("--worker_counts", type=int, nargs="+", default=[1, 2, 4],
                        help="List of worker-process counts to test, e.g. --worker_counts 1 2 4")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"],
                        help="Used for every worker unless --devices is given.")
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated device list, one per worker slot "
                            "(cycled if shorter than the worker count), e.g. "
                            "'cuda:0,cuda:1'. Overrides --device.")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip the 1-worker baseline run (e.g. if already known).")
    parser.add_argument("--output_dir", type=str, default="experiments/scalability")

    return parser.parse_args()


# --------------------------------------------------
# Worker
# --------------------------------------------------
def _worker(job):
    """Runs in a separate process. Loads its own model copy and processes
    only its assigned slice of tile indices. Must be a top-level function
    (not a closure) so it can be pickled by multiprocessing."""
    (rank, device_str, model_path, arch, in_channels, input_path,
    tile_size_px, overlap, normalize_mode, batch_size, indices) = job

    device = resolve_device(device_str)
    model, device = load_model(model_path, arch, in_channels, device)

    dataset = RasterWindowDataset(input_path, tile_size_px, overlap, in_channels, normalize_mode)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    n_pixels = 0
    for idxs, tensors in loader:
        probs = predict_batch(model, tensors, device)
        n_pixels += int(probs.shape[0]) * int(probs.shape[1]) * int(probs.shape[2])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    return {
        "rank": rank, "device": device_str, "n_tiles": len(indices),
        "n_tile_pixels": n_pixels, "elapsed_s": elapsed,
    }


def split_indices(n_items, n_workers):
    """Interleaved split (round-robin) so each worker gets a similar mix of
    tile positions rather than one worker getting all the empty/edge tiles."""
    return [list(range(i, n_items, n_workers)) for i in range(n_workers)]


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()

    device_list = (
        [d.strip() for d in args.devices.split(",")] if args.devices else [args.device]
    )

    # Only need the dataset here to know how many tile windows the raster
    # produces; workers will each open their own RasterWindowDataset.
    probe_ds = RasterWindowDataset(
        args.input, args.tile_size_px, args.overlap, args.in_channels, args.normalize
    )
    n_windows = len(probe_ds)
    all_indices = list(range(n_windows))
    print(f"Raster produces {n_windows} tiles of {args.tile_size_px}px "
        f"(overlap {args.overlap}px).")

    results = {"n_windows": n_windows, "baseline": None, "runs": []}

    baseline_time = None
    if not args.skip_baseline:
        print("\nRunning 1-worker baseline over all tiles...")
        job = (0, device_list[0], args.model_path, args.arch, args.in_channels,
            args.input, args.tile_size_px, args.overlap, args.normalize,
            args.batch_size, all_indices)
        base = _worker(job)
        baseline_time = base["elapsed_s"]
        results["baseline"] = base
        print(f"  baseline: {n_windows} tiles in {baseline_time:.2f}s "
            f"({n_windows / baseline_time:.2f} tiles/s)")

    ctx = mp.get_context("spawn")

    for n_workers in args.worker_counts:
        if n_workers == 1 and baseline_time is not None:
            # Re-use the baseline instead of re-running an identical 1-worker job.
            run_summary = {
                "n_workers": 1, "wall_time_s": baseline_time,
                "speedup_vs_1worker": 1.0, "parallel_efficiency": 1.0,
                "per_worker": [results["baseline"]],
            }
            results["runs"].append(run_summary)
            print(f"[{1:>2} worker ] wall={baseline_time:7.2f}s  speedup=1.00x  efficiency=100%")
            continue

        devices_for_run = [device_list[i % len(device_list)] for i in range(n_workers)]
        chunks = split_indices(n_windows, n_workers)
        jobs = [
            (r, devices_for_run[r], args.model_path, args.arch, args.in_channels,
            args.input, args.tile_size_px, args.overlap, args.normalize,
            args.batch_size, chunks[r])
            for r in range(n_workers)
        ]

        t_start = time.perf_counter()
        with ctx.Pool(processes=n_workers) as pool:
            per_worker = pool.map(_worker, jobs)
        wall_time = time.perf_counter() - t_start

        speedup = (baseline_time / wall_time) if baseline_time else None
        efficiency = (speedup / n_workers) if speedup is not None else None

        run_summary = {
            "n_workers": n_workers,
            "wall_time_s": wall_time,
            "speedup_vs_1worker": speedup,
            "parallel_efficiency": efficiency,
            "per_worker": per_worker,
        }
        results["runs"].append(run_summary)

        msg = f"[{n_workers:>2} workers] wall={wall_time:7.2f}s"
        if speedup is not None:
            msg += f"  speedup={speedup:5.2f}x  efficiency={efficiency * 100:5.1f}%"
        print(msg)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "parallel_scalability_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results -> {json_path}")

    csv_path = out_dir / "parallel_scalability_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["n_workers", "wall_time_s", "speedup_vs_1worker", "parallel_efficiency"]
        )
        writer.writeheader()
        for run in results["runs"]:
            writer.writerow({
                "n_workers": run["n_workers"],
                "wall_time_s": round(run["wall_time_s"], 3),
                "speedup_vs_1worker": round(run["speedup_vs_1worker"], 3) if run["speedup_vs_1worker"] else None,
                "parallel_efficiency": round(run["parallel_efficiency"], 3) if run["parallel_efficiency"] else None,
            })
    print(f"Saved summary table -> {csv_path}")

    # --------------------------------------------------
    # Plot: speedup vs. n_workers (with ideal-linear reference line)
    # --------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        ws = [r["n_workers"] for r in results["runs"]]
        sp = [r["speedup_vs_1worker"] for r in results["runs"]]
        if all(s is not None for s in sp):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(ws, sp, "o-", label="Measured speedup")
            ax.plot(ws, ws, "k--", alpha=0.5, label="Ideal linear speedup")
            ax.set_xlabel("Number of worker processes")
            ax.set_ylabel("Speedup vs. 1 worker")
            ax.set_title("Parallel scalability")
            ax.legend()
            fig.tight_layout()
            plot_path = out_dir / "parallel_scalability_plot.png"
            fig.savefig(plot_path, dpi=150)
            print(f"Saved plot -> {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
