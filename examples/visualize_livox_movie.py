"""Visualize Livox LiDAR point cloud data as animated movie."""

import argparse
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.animation import FuncAnimation


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


def load_all_point_clouds(
    zarr_path: Path, max_frames: int | None = None
) -> tuple[zarr.Group, np.ndarray, int]:
    """Load point cloud zarr group and timestamps.

    Returns:
        Tuple of (zarr_group, timestamps, num_frames).
    """
    group = zarr.open_group(str(zarr_path), mode="r")
    timestamps = np.array(group["timestamp"])

    if max_frames is not None:
        timestamps = timestamps[:max_frames]
        num_frames = max_frames
    else:
        num_frames = len(timestamps)

    return group, timestamps, num_frames


def calculate_fps(timestamps: np.ndarray) -> float:
    """Calculate FPS from timestamps."""
    if len(timestamps) > 1:
        dt = np.diff(timestamps[:100])
        fps = 1.0 / np.mean(dt)
        return fps
    return 10.0


def get_global_bounds(
    group: zarr.Group, num_frames: int, stride: int = 10
) -> tuple[float, float, float, float, float, float]:
    """Calculate global coordinate bounds across frames."""
    print(f"Calculating global bounds (sampling every {stride} frames)...")

    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")
    z_min, z_max = float("inf"), float("-inf")

    for i in range(0, num_frames, stride):
        points = np.array(group["points"][i])
        x_min = min(x_min, points[:, 0].min())
        x_max = max(x_max, points[:, 0].max())
        y_min = min(y_min, points[:, 1].min())
        y_max = max(y_max, points[:, 1].max())
        z_min = min(z_min, points[:, 2].min())
        z_max = max(z_max, points[:, 2].max())

    return x_min, x_max, y_min, y_max, z_min, z_max


def create_lidar_animation(
    group: zarr.Group,
    timestamps: np.ndarray,
    num_frames: int,
    fps: float,
    mission: str,
) -> tuple[plt.Figure, FuncAnimation]:
    """Create animated LiDAR point cloud visualization."""
    # Calculate global bounds for consistent view
    x_min, x_max, y_min, y_max, z_min, z_max = get_global_bounds(group, num_frames)

    # Create figure and 3D axis
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Load first frame
    points = np.array(group["points"][0])
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Create scatter plot
    scatter = ax.scatter(x, y, z, c=z, cmap="viridis", s=0.5, alpha=0.6)

    # Set labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Set fixed bounds
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("Height (m)")
    scatter.set_clim(z_min, z_max)

    # Initial title
    start_time = timestamps[0]

    def update_frame(frame_idx: int) -> tuple:
        """Update function for animation."""
        # Load new point cloud
        points = np.array(group["points"][frame_idx])
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Update scatter data
        scatter._offsets3d = (x, y, z)
        scatter.set_array(z)

        # Update title with frame and timestamp info
        timestamp = timestamps[frame_idx]
        elapsed_time = timestamp - start_time

        ax.set_title(
            f"Livox LiDAR Point Cloud - {mission}\n"
            f"Frame: {frame_idx:04d}/{num_frames} | "
            f"Time: {elapsed_time:.2f}s | "
            f"Timestamp: {timestamp:.3f}s | "
            f"Points: {len(points):,}",
            fontsize=11,
        )

        return (scatter,)

    # Set initial title
    update_frame(0)

    # Calculate interval for real-time playback
    interval_ms = int(1000 / fps)

    anim = FuncAnimation(
        fig, update_frame, frames=num_frames, interval=interval_ms, blit=False
    )

    plt.tight_layout()
    return fig, anim


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize Livox LiDAR as animated movie"
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Lite mode: only display first 500 frames",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to visualize Livox LiDAR movie."""
    args = parse_args()
    mission = "2024-10-01-11-29-55"
    cache_dir = "./cache"
    max_frames = 500 if args.lite else None

    print("=" * 70)
    mode = "Lite Mode (500 frames)" if args.lite else "Full Mode"
    print(f"Livox LiDAR Point Cloud Movie - {mode}")
    print("=" * 70)

    # Load Livox LiDAR tar
    print(f"\nDownloading Livox LiDAR data from mission: {mission}")
    tar_path = load_livox_data(mission, cache_dir)
    print(f"Downloaded: {tar_path.name}")

    # Extract zarr data
    zarr_path = extract_livox_zarr(tar_path, cache_dir)
    print(f"Extracted to: {zarr_path}")

    # Load all point clouds
    print("\nLoading point cloud data...")
    group, timestamps, num_frames = load_all_point_clouds(zarr_path, max_frames)
    print(f"Loaded {num_frames} frames")

    # Calculate FPS
    fps = calculate_fps(timestamps)
    print(f"Measured FPS: {fps:.2f}")

    total_time = timestamps[-1] - timestamps[0]
    print(f"Total duration: {total_time:.1f}s")

    # Create animation
    print("\nCreating animation...")
    fig, anim = create_lidar_animation(group, timestamps, num_frames, fps, mission)

    print("\nDisplaying LiDAR movie (real-time)...")
    print("Close the window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
