"""Export ANYmal joint positions from Zarr to JSON for frontend playback."""

import json
from pathlib import Path

import numpy as np
import zarr

JOINT_NAMES = [
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
]

ZARR_PATH = Path("cache/extracted/anymal_state_actuator/anymal_state_actuator")
OUTPUT_PATH = Path("frontend/public/joint_data.json")
TARGET_FPS = 30


def load_positions(zarr_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load joint positions and timestamps from Zarr store."""
    group = zarr.open_group(str(zarr_path), mode="r")
    timestamps = np.array(group["timestamp"])
    positions = np.zeros((len(timestamps), 12))
    for i in range(12):
        key = f"{i:02d}_state_joint_position"
        positions[:, i] = np.array(group[key])
    return positions, timestamps


def downsample(
    positions: np.ndarray,
    timestamps: np.ndarray,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample data to target FPS by uniform index selection."""
    duration = timestamps[-1] - timestamps[0]
    n_out = int(duration * target_fps)
    indices = np.linspace(0, len(timestamps) - 1, n_out, dtype=int)
    return positions[indices], timestamps[indices]


def export(zarr_path: Path, output_path: Path, fps: float) -> None:
    """Load, downsample, and write joint data to JSON."""
    print(f"Loading from {zarr_path} ...")
    positions, timestamps = load_positions(zarr_path)
    print(f"  {len(timestamps):,} samples loaded")

    positions, timestamps = downsample(positions, timestamps, fps)
    t0 = float(timestamps[0])
    rel_timestamps = (timestamps - t0).tolist()
    print(f"  Downsampled to {len(rel_timestamps):,} frames at {fps} fps")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "joint_names": JOINT_NAMES,
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
