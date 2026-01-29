"""Visualize accumulated Livox LiDAR point clouds with robot pose transformation."""

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


def extract_livox_zarr(tar_path: Path, cache_dir: str) -> Path:
    """Extract Livox zarr data from tar."""
    extract_dir = Path(cache_dir) / "extracted" / "livox_points"

    if not extract_dir.exists():
        print("Extracting Livox LiDAR data...")
        extract_tar_file(tar_path, extract_dir)

    zarr_path = extract_dir / "livox_points"
    return zarr_path


def extract_odometry_zarr(tar_path: Path, cache_dir: str) -> Path:
    """Extract odometry zarr data from tar."""
    extract_dir = Path(cache_dir) / "extracted" / "anymal_state_odometry"

    if not extract_dir.exists():
        print("Extracting odometry data...")
        extract_tar_file(tar_path, extract_dir)

    zarr_path = extract_dir / "anymal_state_odometry"
    return zarr_path


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = quat

    # Compute rotation matrix elements
    r00 = 1 - 2 * (qy**2 + qz**2)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)

    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx**2 + qz**2)
    r12 = 2 * (qy * qz - qx * qw)

    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx**2 + qy**2)

    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])


def transform_points(
    points: np.ndarray, position: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    """Transform points by robot pose (position + orientation).

    Args:
        points: Nx3 array of points
        position: 3-element array [x, y, z]
        orientation: 4-element quaternion [qx, qy, qz, qw]

    Returns:
        Nx3 array of transformed points
    """
    rotation_matrix = quaternion_to_rotation_matrix(orientation)
    transformed = (rotation_matrix @ points.T).T + position
    return transformed


def load_synchronized_data(
    lidar_zarr: Path, odom_zarr: Path, max_frames: int | None = None
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Load and synchronize LiDAR and odometry data.

    Returns:
        Tuple of (lidar_frames, lidar_timestamps, positions, orientations)
    """
    lidar_group = zarr.open_group(str(lidar_zarr), mode="r")
    odom_group = zarr.open_group(str(odom_zarr), mode="r")

    lidar_timestamps = np.array(lidar_group["timestamp"])
    odom_timestamps = np.array(odom_group["timestamp"])
    positions = np.array(odom_group["pose_pos"])
    orientations = np.array(odom_group["pose_orien"])

    if max_frames is not None:
        num_frames = min(max_frames, len(lidar_timestamps))
    else:
        num_frames = len(lidar_timestamps)

    lidar_frames = []
    synced_positions = []
    synced_orientations = []
    synced_timestamps = []

    for i in range(num_frames):
        lidar_t = lidar_timestamps[i]

        # Find closest odometry frame
        time_diffs = np.abs(odom_timestamps - lidar_t)
        closest_idx = np.argmin(time_diffs)

        # Only use if time difference is reasonable (< 0.1s)
        if time_diffs[closest_idx] < 0.1:
            points = np.array(lidar_group["points"][i])
            lidar_frames.append(points)
            synced_positions.append(positions[closest_idx])
            synced_orientations.append(orientations[closest_idx])
            synced_timestamps.append(lidar_t)

    return (
        lidar_frames,
        np.array(synced_timestamps),
        np.array(synced_positions),
        np.array(synced_orientations),
    )


def create_accumulated_animation(
    lidar_frames: list[np.ndarray],
    timestamps: np.ndarray,
    positions: np.ndarray,
    orientations: np.ndarray,
    mission: str,
    fps: float,
    downsample: int = 10,
    mode_2d: bool = False,
) -> tuple[plt.Figure, FuncAnimation]:
    """Create animation showing accumulated LiDAR points.

    Args:
        downsample: Only plot every Nth point for performance
        mode_2d: If True, show top-down 2D view (XY plane)
    """
    fig = plt.figure(figsize=(14, 10))
    if mode_2d:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection="3d")

    # Accumulate all transformed points
    all_points = []
    all_colors = []

    def update_frame(frame_idx: int) -> tuple:
        """Update function for animation."""
        # Transform and add new points from this frame
        if frame_idx < len(lidar_frames):
            points = lidar_frames[frame_idx]
            position = positions[frame_idx]
            orientation = orientations[frame_idx]

            # Transform points to world frame
            transformed = transform_points(points, position, orientation)

            # Downsample for performance
            if downsample > 1:
                transformed = transformed[::downsample]

            all_points.append(transformed)

            # Color by frame index (time progression)
            color_val = frame_idx / len(lidar_frames)
            colors = np.full(len(transformed), color_val)
            all_colors.append(colors)

        # Clear and replot all accumulated points
        ax.clear()

        if all_points:
            # Concatenate all points
            combined_points = np.vstack(all_points)
            combined_colors = np.concatenate(all_colors)

            if mode_2d:
                # Plot accumulated point cloud (XY plane only)
                ax.scatter(
                    combined_points[:, 0],
                    combined_points[:, 1],
                    c=combined_colors,
                    cmap="viridis",
                    s=0.5,
                    alpha=0.6,
                )

                # Plot robot trajectory
                traj_positions = positions[: frame_idx + 1]
                ax.plot(
                    traj_positions[:, 0],
                    traj_positions[:, 1],
                    "r-",
                    linewidth=2,
                    alpha=0.8,
                    label="Robot Path",
                )

                # Plot current robot position
                if frame_idx < len(positions):
                    ax.scatter(
                        positions[frame_idx, 0],
                        positions[frame_idx, 1],
                        c="red",
                        s=100,
                        marker="o",
                        label="Robot",
                    )
            else:
                # Plot accumulated point cloud (3D)
                ax.scatter(
                    combined_points[:, 0],
                    combined_points[:, 1],
                    combined_points[:, 2],
                    c=combined_colors,
                    cmap="viridis",
                    s=0.1,
                    alpha=0.6,
                )

                # Plot robot trajectory
                traj_positions = positions[: frame_idx + 1]
                ax.plot(
                    traj_positions[:, 0],
                    traj_positions[:, 1],
                    traj_positions[:, 2],
                    "r-",
                    linewidth=2,
                    alpha=0.8,
                    label="Robot Path",
                )

                # Plot current robot position
                if frame_idx < len(positions):
                    ax.scatter(
                        positions[frame_idx, 0],
                        positions[frame_idx, 1],
                        positions[frame_idx, 2],
                        c="red",
                        s=100,
                        marker="o",
                        label="Robot",
                    )

        # Set labels and title
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        if not mode_2d:
            ax.set_zlabel("Z (m)")

        elapsed_time = (
            timestamps[frame_idx] - timestamps[0] if frame_idx < len(timestamps) else 0
        )
        view_mode = "2D Top-Down (XY Plane)" if mode_2d else "3D"
        ax.set_title(
            f"Accumulated LiDAR Point Cloud - {mission} ({view_mode})\n"
            f"Frame: {frame_idx + 1}/{len(lidar_frames)} | "
            f"Time: {elapsed_time:.2f}s | "
            f"Total Points: {len(all_points) * len(lidar_frames[0]) // downsample if all_points else 0:,}",
            fontsize=11,
        )

        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        if mode_2d:
            ax.set_aspect("equal")

        return (ax,)

    # Calculate interval
    interval_ms = int(1000 / fps)

    anim = FuncAnimation(
        fig, update_frame, frames=len(lidar_frames), interval=interval_ms, blit=False
    )

    plt.tight_layout()
    return fig, anim


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize accumulated LiDAR point cloud with robot pose"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Maximum number of frames to process (default: 50)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=10,
        help="Plot every Nth point for performance (default: 10)",
    )
    parser.add_argument(
        "--fps", type=float, default=5.0, help="Animation FPS (default: 5.0)"
    )
    parser.add_argument(
        "--2d",
        dest="mode_2d",
        action="store_true",
        help="Show 2D top-down view (XY plane)",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to visualize accumulated LiDAR."""
    args = parse_args()
    mission = "2024-10-01-11-29-55"
    cache_dir = "./cache"

    print("=" * 60)
    print("Accumulated LiDAR Point Cloud Visualization")
    print("=" * 60)
    print(f"Max frames: {args.max_frames}")
    print(f"Downsample factor: {args.downsample}")
    print(f"Animation FPS: {args.fps}")
    print(f"View mode: {'2D (XY plane)' if args.mode_2d else '3D'}")

    # Load LiDAR data
    print(f"\nDownloading LiDAR data from mission: {mission}")
    lidar_tar = load_livox_data(mission, cache_dir)
    print(f"Downloaded: {lidar_tar.name}")

    # Load odometry data
    print(f"\nDownloading odometry data from mission: {mission}")
    odom_tar = load_odometry_data(mission, cache_dir)
    print(f"Downloaded: {odom_tar.name}")

    # Extract zarr data
    lidar_zarr = extract_livox_zarr(lidar_tar, cache_dir)
    odom_zarr = extract_odometry_zarr(odom_tar, cache_dir)
    print(f"\nExtracted LiDAR: {lidar_zarr}")
    print(f"Extracted odometry: {odom_zarr}")

    # Load and synchronize data
    print("\nLoading and synchronizing LiDAR and odometry data...")
    lidar_frames, timestamps, positions, orientations = load_synchronized_data(
        lidar_zarr, odom_zarr, args.max_frames
    )
    print(f"Loaded {len(lidar_frames)} synchronized frames")

    # Create animation
    print("\nCreating accumulated point cloud animation...")
    fig, anim = create_accumulated_animation(
        lidar_frames,
        timestamps,
        positions,
        orientations,
        mission,
        args.fps,
        args.downsample,
        args.mode_2d,
    )

    print("\nDisplaying animation (close window to exit)...")
    print("Note: Animation may take time to render each frame.")
    plt.show()


if __name__ == "__main__":
    main()
