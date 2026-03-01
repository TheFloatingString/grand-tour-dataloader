"""Export ANYmal base position from odometry Zarr to JSON for frontend."""

import json
from pathlib import Path

import numpy as np
import zarr

ZARR_PATH = Path("cache/extracted/anymal_state_odometry/anymal_state_odometry")
OUTPUT_PATH = Path("frontend/public/base_position.json")
TARGET_FPS = 30


def load_odometry(zarr_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load base position and timestamps from Zarr store."""
    group = zarr.open_group(str(zarr_path), mode="r")
    timestamps = np.array(group["timestamp"])
    positions = np.array(group["pose_pos"])  # Nx3: [x, y, z]
    return positions, timestamps


def downsample(
    positions: np.ndarray,
    timestamps: np.ndarray,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample to target FPS by uniform index selection."""
    duration = timestamps[-1] - timestamps[0]
    n_out = int(duration * target_fps)
    indices = np.linspace(0, len(timestamps) - 1, n_out, dtype=int)
    return positions[indices], timestamps[indices]


def export(zarr_path: Path, output_path: Path, fps: float) -> None:
    """Load, downsample, and write base position data to JSON."""
    print(f"Loading from {zarr_path} ...")
    positions, timestamps = load_odometry(zarr_path)
    print(f"  {len(timestamps):,} samples loaded")

    positions, timestamps = downsample(positions, timestamps, fps)
    t0 = float(timestamps[0])
    rel_timestamps = (timestamps - t0).tolist()
    print(f"  Downsampled to {len(rel_timestamps):,} frames at {fps} fps")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fps": fps,
        "duration": rel_timestamps[-1],
        "timestamps": rel_timestamps,
        "frames": positions.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f)
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Written to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    export(ZARR_PATH, OUTPUT_PATH, TARGET_FPS)
