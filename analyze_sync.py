"""Analyze timestamp synchronization across three front cameras."""

import tarfile
from pathlib import Path

import numpy as np
import zarr


def extract_camera_data(tar_path: Path, extract_dir: Path) -> None:
    """Extract camera data tar to directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(extract_dir, filter="data")


def load_camera_timestamps(camera_name: str, cache_dir: str) -> np.ndarray:
    """Load timestamps for a camera sensor."""
    data_tar_path = (
        Path(cache_dir) / "datasets--leggedrobotics--grand_tour_dataset/snapshots"
    )
    snapshot_dirs = list(data_tar_path.glob("*/"))

    if not snapshot_dirs:
        return np.array([])

    # Extract tar if not already done
    tar_path = snapshot_dirs[0] / "2024-10-01-11-29-55" / "data" / f"{camera_name}.tar"
    extract_dir = Path(cache_dir) / "extracted" / camera_name

    if not extract_dir.exists():
        print(f"Extracting {camera_name} data tar...")
        extract_camera_data(tar_path, extract_dir)

    # Load timestamps from zarr
    zarr_path = extract_dir / camera_name
    try:
        group = zarr.open_group(str(zarr_path), mode="r")
        return np.array(group["timestamp"])
    except Exception as e:
        print(f"Error loading {camera_name}: {e}")
        return np.array([])


def analyze_synchronization(all_timestamps: dict[str, np.ndarray]) -> None:
    """Analyze timestamp synchronization across cameras."""
    # Check first 10 frames
    print("\n" + "=" * 70)
    print("First 10 frames - Absolute timestamps and offsets:")
    print("=" * 70)

    for i in range(10):
        left_ts = all_timestamps["alphasense_front_left"][i]
        center_ts = all_timestamps["alphasense_front_center"][i]
        right_ts = all_timestamps["alphasense_front_right"][i]

        delta_lc = (center_ts - left_ts) * 1000  # ms
        delta_cr = (right_ts - center_ts) * 1000  # ms

        print(f"\nFrame {i}:")
        print(f"  Left:   {left_ts:.6f}s")
        print(f"  Center: {center_ts:.6f}s  (offset from Left:   {delta_lc:+.2f}ms)")
        print(f"  Right:  {right_ts:.6f}s  (offset from Center: {delta_cr:+.2f}ms)")

    # Statistical analysis
    print("\n" + "=" * 70)
    print("Statistical Analysis of Camera Synchronization:")
    print("=" * 70)

    min_len = min(len(ts) for ts in all_timestamps.values())

    left_ts = all_timestamps["alphasense_front_left"][:min_len]
    center_ts = all_timestamps["alphasense_front_center"][:min_len]
    right_ts = all_timestamps["alphasense_front_right"][:min_len]

    delta_lc = (center_ts - left_ts) * 1000  # ms
    delta_cr = (right_ts - center_ts) * 1000  # ms
    delta_lr = (right_ts - left_ts) * 1000  # ms

    print("\nCenter - Left offset:")
    print(f"  Mean: {np.mean(delta_lc):.2f}ms")
    print(f"  Std:  {np.std(delta_lc):.2f}ms")
    print(f"  Min:  {np.min(delta_lc):.2f}ms")
    print(f"  Max:  {np.max(delta_lc):.2f}ms")

    print("\nRight - Center offset:")
    print(f"  Mean: {np.mean(delta_cr):.2f}ms")
    print(f"  Std:  {np.std(delta_cr):.2f}ms")
    print(f"  Min:  {np.min(delta_cr):.2f}ms")
    print(f"  Max:  {np.max(delta_cr):.2f}ms")

    print("\nRight - Left offset:")
    print(f"  Mean: {np.mean(delta_lr):.2f}ms")
    print(f"  Std:  {np.std(delta_lr):.2f}ms")
    print(f"  Min:  {np.min(delta_lr):.2f}ms")
    print(f"  Max:  {np.max(delta_lr):.2f}ms")

    # Check for consistent offset (hardware synchronized)
    if np.std(delta_lc) < 0.5 and np.std(delta_cr) < 0.5:
        print("\n[OK] Cameras appear to be hardware-synchronized")
        print("  (Temporal offsets are consistent with <0.5ms variance)")
    else:
        print("\n[WARNING] Cameras may not be hardware-synchronized")
        print("  (Temporal offsets show high variance)")

    # Interpretation
    print("\n" + "=" * 70)
    print("Interpretation:")
    print("=" * 70)
    if abs(np.mean(delta_lc) - 100.0) < 1.0:
        print("\nLeft and Center cameras:")
        print("  - Consistent 100ms offset (Center leads by 100ms)")
        print("  - Likely sequential triggering in L -> C order")

    if np.std(delta_cr) > 10.0:
        print("\nCenter and Right cameras:")
        print("  - Variable offset pattern detected")
        print("  - Suggests right camera may have different trigger timing")
    else:
        print("\nCenter and Right cameras:")
        print(f"  - Consistent {np.mean(delta_cr):.2f}ms offset")
        print("  - Hardware synchronized")


def main() -> None:
    """Main analysis function."""
    cache_dir = "./cache"

    cameras = [
        "alphasense_front_left",
        "alphasense_front_center",
        "alphasense_front_right",
    ]

    print("=" * 70)
    print("Camera Timestamp Synchronization Analysis")
    print("=" * 70)

    all_timestamps = {}
    for camera in cameras:
        print(f"\nLoading {camera} timestamps...")
        ts = load_camera_timestamps(camera, cache_dir)
        all_timestamps[camera] = ts
        print(f"  Loaded {len(ts)} timestamps")

    if all(len(ts) > 0 for ts in all_timestamps.values()):
        analyze_synchronization(all_timestamps)
    else:
        print("\nError: Could not load timestamps from all cameras")


if __name__ == "__main__":
    main()
