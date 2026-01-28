"""Visualize ANYmal odometry trajectory."""

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


def load_odometry_data(mission: str, cache_dir: str) -> Path:
    """Load ANYmal odometry data from mission."""
    from huggingface_hub import snapshot_download

    allow_patterns = [f"{mission}/data/anymal_state_odometry.tar"]

    cache_path = snapshot_download(
        repo_id="leggedrobotics/grand_tour_dataset",
        allow_patterns=allow_patterns,
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    data_path = Path(cache_path) / mission / "data" / "anymal_state_odometry.tar"
    return data_path


def extract_odometry_zarr(tar_path: Path, cache_dir: str) -> Path:
    """Extract odometry zarr data from tar."""
    extract_dir = Path(cache_dir) / "extracted" / "anymal_state_odometry"

    if not extract_dir.exists():
        print("Extracting odometry data...")
        extract_tar_file(tar_path, extract_dir)

    zarr_path = extract_dir / "anymal_state_odometry"
    return zarr_path


def load_odometry_trajectory(
    zarr_path: Path, max_frames: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load odometry trajectory data.

    Returns:
        Tuple of (positions, timestamps, orientations).
        positions: Nx3 array of [x, y, z] positions
        timestamps: N array of timestamps
        orientations: Nx4 array of [qx, qy, qz, qw] quaternions
    """
    group = zarr.open_group(str(zarr_path), mode="r")

    # Load position data (abbreviated field names)
    pose_position = np.array(group["pose_pos"])
    timestamps = np.array(group["timestamp"])

    # Load orientation if available
    if "pose_orien" in group:
        pose_orientation = np.array(group["pose_orien"])
    else:
        pose_orientation = None

    if max_frames is not None:
        pose_position = pose_position[:max_frames]
        timestamps = timestamps[:max_frames]
        if pose_orientation is not None:
            pose_orientation = pose_orientation[:max_frames]

    return pose_position, timestamps, pose_orientation


def print_trajectory_stats(positions: np.ndarray, timestamps: np.ndarray) -> None:
    """Print statistics about the trajectory."""
    print("\n" + "=" * 60)
    print("Odometry Trajectory Statistics")
    print("=" * 60)

    total_time = timestamps[-1] - timestamps[0]
    print(f"Total points: {len(positions):,}")
    print(f"Duration: {total_time:.2f}s")
    print(f"Sample rate: {len(positions) / total_time:.2f} Hz")

    # Calculate total distance traveled
    deltas = np.diff(positions, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    total_distance = np.sum(distances)

    print(f"\nTotal distance traveled: {total_distance:.2f}m")
    print(f"Average speed: {total_distance / total_time:.2f} m/s")

    print("\nPosition ranges:")
    print(f"  X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m")
    print(f"  Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m")
    print(f"  Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m")


def visualize_trajectory_2d(
    positions: np.ndarray, timestamps: np.ndarray, mission: str
) -> None:
    """Visualize trajectory in 2D top-down view."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left plot: 2D trajectory (top-down view)
    x, y = positions[:, 0], positions[:, 1]

    # Color by time
    elapsed_time = timestamps - timestamps[0]
    scatter = ax1.scatter(x, y, c=elapsed_time, cmap="viridis", s=2, alpha=0.6)

    # Mark start and end
    ax1.plot(x[0], y[0], "go", markersize=10, label="Start")
    ax1.plot(x[-1], y[-1], "ro", markersize=10, label="End")

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title(f"ANYmal Trajectory (Top-Down)\n{mission}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("Time (s)")

    # Right plot: Height profile over time
    z = positions[:, 2]
    ax2.plot(elapsed_time, z, linewidth=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Height Z (m)")
    ax2.set_title("Height Profile Over Time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()


def visualize_trajectory_3d(
    positions: np.ndarray, timestamps: np.ndarray, mission: str
) -> None:
    """Visualize trajectory in 3D."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # Color by time
    elapsed_time = timestamps - timestamps[0]
    scatter = ax.scatter(x, y, z, c=elapsed_time, cmap="viridis", s=2, alpha=0.6)

    # Mark start and end
    ax.scatter(x[0], y[0], z[0], c="green", s=100, marker="o", label="Start")
    ax.scatter(x[-1], y[-1], z[-1], c="red", s=100, marker="o", label="End")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"ANYmal 3D Trajectory\n{mission}")
    ax.legend()

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Time (s)")

    plt.tight_layout()


def create_trajectory_animation(
    positions: np.ndarray, timestamps: np.ndarray, mission: str, fps: float
) -> tuple[plt.Figure, FuncAnimation]:
    """Create animated trajectory visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    start_time = timestamps[0]

    # Initialize plots
    (line1,) = ax1.plot([], [], "b-", linewidth=1, alpha=0.6)
    (point1,) = ax1.plot([], [], "ro", markersize=8)
    ax1.plot(x[0], y[0], "go", markersize=10, label="Start")

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_xlim(x.min() - 5, x.max() + 5)
    ax1.set_ylim(y.min() - 5, y.max() + 5)
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")
    ax1.legend()

    # Height profile
    (line2,) = ax2.plot([], [], "b-", linewidth=1)
    (point2,) = ax2.plot([], [], "ro", markersize=8)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Height Z (m)")
    ax2.set_xlim(0, timestamps[-1] - timestamps[0])
    ax2.set_ylim(z.min() - 1, z.max() + 1)
    ax2.grid(True, alpha=0.3)

    def update_frame(frame_idx: int) -> tuple:
        """Update function for animation."""
        # Update trajectory
        line1.set_data(x[: frame_idx + 1], y[: frame_idx + 1])
        point1.set_data([x[frame_idx]], [y[frame_idx]])

        # Update height profile
        elapsed = timestamps[: frame_idx + 1] - start_time
        line2.set_data(elapsed, z[: frame_idx + 1])
        point2.set_data([elapsed[frame_idx]], [z[frame_idx]])

        # Update title
        timestamp = timestamps[frame_idx]
        elapsed_time = timestamp - start_time

        # Calculate distance traveled
        if frame_idx > 0:
            deltas = np.diff(positions[: frame_idx + 1], axis=0)
            distances = np.linalg.norm(deltas, axis=1)
            total_dist = np.sum(distances)
        else:
            total_dist = 0.0

        fig.suptitle(
            f"ANYmal Odometry - {mission}\n"
            f"Frame: {frame_idx:05d}/{len(positions)} | "
            f"Time: {elapsed_time:.2f}s | "
            f"Timestamp: {timestamp:.3f}s | "
            f"Distance: {total_dist:.2f}m",
            fontsize=11,
        )

        return line1, point1, line2, point2

    # Set initial title
    update_frame(0)

    # Calculate interval
    interval_ms = int(1000 / fps)

    anim = FuncAnimation(
        fig, update_frame, frames=len(positions), interval=interval_ms, blit=False
    )

    plt.tight_layout()
    return fig, anim


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize ANYmal odometry trajectory")
    parser.add_argument(
        "--mode",
        choices=["2d", "3d", "movie"],
        default="2d",
        help="Visualization mode (default: 2d)",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Lite mode: only display first 10000 points",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to visualize odometry."""
    args = parse_args()
    mission = "2024-10-01-11-29-55"
    cache_dir = "./cache"
    max_frames = 10000 if args.lite else None

    print("=" * 60)
    mode_name = args.mode.upper()
    lite_str = " (Lite Mode)" if args.lite else ""
    print(f"ANYmal Odometry Visualization - {mode_name}{lite_str}")
    print("=" * 60)

    # Load odometry tar
    print(f"\nDownloading odometry data from mission: {mission}")
    tar_path = load_odometry_data(mission, cache_dir)
    print(f"Downloaded: {tar_path.name}")

    # Extract zarr data
    zarr_path = extract_odometry_zarr(tar_path, cache_dir)
    print(f"Extracted to: {zarr_path}")

    # Load trajectory
    print("\nLoading trajectory data...")
    positions, timestamps, orientations = load_odometry_trajectory(
        zarr_path, max_frames
    )
    print(f"Loaded {len(positions)} odometry points")

    # Print statistics
    print_trajectory_stats(positions, timestamps)

    # Visualize based on mode
    print(f"\nCreating {args.mode} visualization...")

    if args.mode == "2d":
        visualize_trajectory_2d(positions, timestamps, mission)
    elif args.mode == "3d":
        visualize_trajectory_3d(positions, timestamps, mission)
    elif args.mode == "movie":
        # Calculate FPS
        if len(timestamps) > 1:
            dt = np.diff(timestamps[:100])
            fps = 1.0 / np.mean(dt)
        else:
            fps = 100.0
        print(f"Measured FPS: {fps:.2f}")
        create_trajectory_animation(positions, timestamps, mission, fps)

    print("\nDisplaying visualization (close window to exit)...")
    plt.show()


if __name__ == "__main__":
    main()
