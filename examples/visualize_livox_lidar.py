"""Visualize Livox LiDAR point cloud data."""

import argparse
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr


def extract_tar_file(tar_path: Path, extract_dir: Path) -> Path:
    """Extract tar file to specified directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(extract_dir, filter="data")
    return extract_dir


def load_livox_data(mission: str, cache_dir: str) -> Path:
    """Load Livox LiDAR data from mission."""
    from huggingface_hub import snapshot_download

    allow_patterns = [f"{mission}/data/livox_points.tar"]

    cache_path = snapshot_download(
        repo_id="leggedrobotics/grand_tour_dataset",
        allow_patterns=allow_patterns,
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    data_path = Path(cache_path) / mission / "data" / "livox_points.tar"
    return data_path


def extract_livox_zarr(tar_path: Path, cache_dir: str) -> Path:
    """Extract Livox zarr data from tar."""
    extract_dir = Path(cache_dir) / "extracted" / "livox_points"

    if not extract_dir.exists():
        print("Extracting Livox LiDAR data...")
        extract_tar_file(tar_path, extract_dir)

    zarr_path = extract_dir / "livox_points"
    return zarr_path


def load_point_cloud_frame(
    zarr_path: Path, frame_idx: int = 0
) -> tuple[np.ndarray, float]:
    """Load a single frame of point cloud data.

    Returns:
        Tuple of (points, timestamp) where points is Nx3 array.
    """
    group = zarr.open_group(str(zarr_path), mode="r")

    # Load point cloud data for the frame
    points = np.array(group["points"][frame_idx])
    timestamp = float(group["timestamp"][frame_idx])

    return points, timestamp


def visualize_point_cloud(points: np.ndarray, timestamp: float, mission: str) -> None:
    """Visualize point cloud in 3D matplotlib plot."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Extract x, y, z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Color by height (z-coordinate)
    colors = z
    scatter = ax.scatter(x, y, z, c=colors, cmap="viridis", s=0.5, alpha=0.6)

    # Set labels and title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"Livox LiDAR Point Cloud\nMission: {mission}\nTimestamp: {timestamp:.3f}s"
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Height (m)")

    # Set equal aspect ratio
    max_range = (
        np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    )
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()


def print_point_cloud_stats(points: np.ndarray, timestamp: float) -> None:
    """Print statistics about the point cloud."""
    print("\n" + "=" * 60)
    print("Point Cloud Statistics")
    print("=" * 60)
    print(f"Timestamp: {timestamp:.6f}s")
    print(f"Number of points: {len(points):,}")
    print("\nCoordinate ranges:")
    print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] m")
    print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] m")
    print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] m")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize Livox LiDAR point cloud")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to visualize (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to visualize Livox LiDAR data."""
    args = parse_args()
    mission = "2024-10-01-11-29-55"
    cache_dir = "./cache"

    print("=" * 60)
    print("Livox LiDAR Point Cloud Visualization")
    print("=" * 60)

    # Load Livox LiDAR tar
    print(f"\nDownloading Livox LiDAR data from mission: {mission}")
    tar_path = load_livox_data(mission, cache_dir)
    print(f"Downloaded: {tar_path.name}")

    # Extract zarr data
    zarr_path = extract_livox_zarr(tar_path, cache_dir)
    print(f"Extracted to: {zarr_path}")

    # Load specific frame
    print(f"\nLoading frame {args.frame}...")
    points, timestamp = load_point_cloud_frame(zarr_path, args.frame)

    # Print statistics
    print_point_cloud_stats(points, timestamp)

    # Visualize
    print("\nCreating 3D visualization...")
    visualize_point_cloud(points, timestamp, mission)

    print("\nDisplaying point cloud (close window to exit)...")
    plt.show()


if __name__ == "__main__":
    main()
