"""Inspect the structure of Livox LiDAR data."""

from pathlib import Path

import numpy as np
import zarr


def inspect_lidar_structure() -> None:
    """Inspect the zarr structure of Livox LiDAR data."""
    zarr_path = Path("./cache/extracted/livox_points/livox_points")

    if not zarr_path.exists():
        print("Error: Livox data not extracted yet.")
        print("Run visualize_livox_lidar.py or visualize_livox_movie.py first.")
        return

    print("=" * 70)
    print("Livox LiDAR Zarr Structure")
    print("=" * 70)

    # Open zarr group
    group = zarr.open_group(str(zarr_path), mode="r")

    print("\nZarr group contents:")
    print(f"  Keys: {list(group.keys())}")

    # Inspect 'points' array
    if "points" in group:
        points_array = group["points"]
        print("\n'points' array:")
        print(f"  Shape: {points_array.shape}")
        print(f"  Dtype: {points_array.dtype}")
        print(f"  Chunks: {points_array.chunks}")
        print(f"  Total frames: {points_array.shape[0]}")
        print(f"  Points per frame: {points_array.shape[1]}")
        print(f"  Dimensions per point: {points_array.shape[2]}")

        # Load one frame
        print("\nExample frame (frame 100):")
        frame_100 = np.array(points_array[100])
        print(f"  Shape: {frame_100.shape}")
        print(f"  First 5 points:")
        for i in range(5):
            x, y, z = frame_100[i]
            print(f"    Point {i}: x={x:7.3f}m, y={y:7.3f}m, z={z:6.3f}m")

    # Inspect 'timestamp' array
    if "timestamp" in group:
        timestamps = group["timestamp"]
        print("\n'timestamp' array:")
        print(f"  Shape: {timestamps.shape}")
        print(f"  Dtype: {timestamps.dtype}")
        print(f"  Total timestamps: {timestamps.shape[0]}")

        # Show some timestamps
        print("\nExample timestamps:")
        for i in [0, 1, 2, 100, 101]:
            if i < timestamps.shape[0]:
                print(f"  Frame {i:3d}: {timestamps[i]:.6f}s")

    # Check for other fields
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("\nLiDAR data is stored as:")
    print("  - 'points': 3D array of shape (num_frames, points_per_frame, 3)")
    print("    - Dimension 0: Frame index (time)")
    print("    - Dimension 1: Point index within frame")
    print("    - Dimension 2: [x, y, z] coordinates in meters")
    print("  - 'timestamp': 1D array of Unix timestamps (seconds)")
    print("\nEach frame is a snapshot of the environment at a specific time,")
    print("represented as ~25,000 (x,y,z) points in 3D space.")


if __name__ == "__main__":
    inspect_lidar_structure()
