"""Visualize ANYmal actuator (joint) data."""

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


def load_actuator_data(mission: str, cache_dir: str) -> Path:
    """Load actuator data from mission."""
    from huggingface_hub import snapshot_download

    allow_patterns = [f"{mission}/data/anymal_state_actuator.tar"]

    cache_path = snapshot_download(
        repo_id="leggedrobotics/grand_tour_dataset",
        allow_patterns=allow_patterns,
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    data_path = Path(cache_path) / mission / "data" / "anymal_state_actuator.tar"
    return data_path


def extract_actuator_zarr(tar_path: Path, cache_dir: str) -> Path:
    """Extract actuator zarr data from tar."""
    extract_dir = Path(cache_dir) / "extracted" / "anymal_state_actuator"

    if not extract_dir.exists():
        print("Extracting actuator data...")
        extract_tar_file(tar_path, extract_dir)

    zarr_path = extract_dir / "anymal_state_actuator"
    return zarr_path


def load_joint_data(
    zarr_path: Path, max_frames: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load joint position, velocity, and torque data.

    Returns:
        Tuple of (positions, velocities, torques, timestamps).
        Each is shape (N, 12) for N timesteps and 12 joints.
    """
    group = zarr.open_group(str(zarr_path), mode="r")

    # Load timestamps
    timestamps = np.array(group["timestamp"])
    num_samples = len(timestamps) if max_frames is None else max_frames

    # Initialize arrays for 12 joints
    positions = np.zeros((num_samples, 12))
    velocities = np.zeros((num_samples, 12))
    torques = np.zeros((num_samples, 12))

    # Load data for each joint
    for joint_id in range(12):
        joint_prefix = f"{joint_id:02d}"
        positions[:, joint_id] = group[f"{joint_prefix}_state_joint_position"][
            :num_samples
        ]
        velocities[:, joint_id] = group[f"{joint_prefix}_state_joint_velocity"][
            :num_samples
        ]
        torques[:, joint_id] = group[f"{joint_prefix}_state_joint_torque"][:num_samples]

    timestamps = timestamps[:num_samples]
    return positions, velocities, torques, timestamps


def print_joint_stats(
    positions: np.ndarray,
    velocities: np.ndarray,
    torques: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    """Print statistics about joint data."""
    print("\n" + "=" * 60)
    print("Actuator Statistics")
    print("=" * 60)

    total_time = timestamps[-1] - timestamps[0]
    print(f"Total samples: {len(timestamps):,}")
    print(f"Duration: {total_time:.2f}s")
    print(f"Sample rate: {len(timestamps) / total_time:.2f} Hz")

    print("\nJoint position ranges (rad):")
    for i in range(12):
        print(
            f"  Joint {i:02d}: [{positions[:, i].min():6.3f}, {positions[:, i].max():6.3f}]"
        )


def visualize_joints_static(
    positions: np.ndarray,
    velocities: np.ndarray,
    torques: np.ndarray,
    timestamps: np.ndarray,
    mission: str,
) -> None:
    """Visualize joint data as static plots."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    elapsed_time = timestamps - timestamps[0]

    # Joint names for ANYmal (4 legs x 3 joints)
    leg_names = ["LF", "RF", "LH", "RH"]
    joint_names_per_leg = ["HAA", "HFE", "KFE"]
    colors = plt.cm.tab10(np.linspace(0, 1, 12))

    # Plot positions
    for i in range(12):
        leg_idx = i // 3
        joint_idx = i % 3
        label = f"{leg_names[leg_idx]}_{joint_names_per_leg[joint_idx]}"
        axes[0].plot(
            elapsed_time,
            positions[:, i],
            label=label,
            alpha=0.7,
            linewidth=0.8,
            color=colors[i],
        )

    axes[0].set_ylabel("Position (rad)")
    axes[0].set_title(f"ANYmal Joint States - {mission}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=4, fontsize=8)

    # Plot velocities
    for i in range(12):
        axes[1].plot(
            elapsed_time, velocities[:, i], alpha=0.7, linewidth=0.8, color=colors[i]
        )

    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].grid(True, alpha=0.3)

    # Plot torques
    for i in range(12):
        axes[2].plot(
            elapsed_time, torques[:, i], alpha=0.7, linewidth=0.8, color=colors[i]
        )

    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()


def create_joint_animation(
    positions: np.ndarray,
    velocities: np.ndarray,
    torques: np.ndarray,
    timestamps: np.ndarray,
    mission: str,
    fps: float,
) -> tuple[plt.Figure, FuncAnimation]:
    """Create animated joint visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    elapsed_time = timestamps - timestamps[0]
    num_samples = len(timestamps)

    # Colors for 12 joints
    colors = plt.cm.tab10(np.linspace(0, 1, 12))
    leg_names = ["LF", "RF", "LH", "RH"]
    joint_names_per_leg = ["HAA", "HFE", "KFE"]

    # Initialize line plots
    lines_pos = []
    lines_vel = []
    lines_torque = []

    for i in range(12):
        leg_idx = i // 3
        joint_idx = i % 3
        label = f"{leg_names[leg_idx]}_{joint_names_per_leg[joint_idx]}"

        (line_p,) = axes[0].plot(
            [], [], label=label, alpha=0.7, linewidth=0.8, color=colors[i]
        )
        (line_v,) = axes[1].plot([], [], alpha=0.7, linewidth=0.8, color=colors[i])
        (line_t,) = axes[2].plot([], [], alpha=0.7, linewidth=0.8, color=colors[i])

        lines_pos.append(line_p)
        lines_vel.append(line_v)
        lines_torque.append(line_t)

    # Set up axes
    axes[0].set_xlim(0, elapsed_time[-1])
    axes[0].set_ylim(positions.min() - 0.2, positions.max() + 0.2)
    axes[0].set_ylabel("Position (rad)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=4, fontsize=8)

    axes[1].set_xlim(0, elapsed_time[-1])
    axes[1].set_ylim(velocities.min() - 1, velocities.max() + 1)
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlim(0, elapsed_time[-1])
    axes[2].set_ylim(torques.min() - 2, torques.max() + 2)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].grid(True, alpha=0.3)

    def update_frame(frame_idx: int) -> list:
        """Update function for animation."""
        # Update all joint lines
        for i in range(12):
            lines_pos[i].set_data(
                elapsed_time[: frame_idx + 1], positions[: frame_idx + 1, i]
            )
            lines_vel[i].set_data(
                elapsed_time[: frame_idx + 1], velocities[: frame_idx + 1, i]
            )
            lines_torque[i].set_data(
                elapsed_time[: frame_idx + 1], torques[: frame_idx + 1, i]
            )

        # Update title
        timestamp = timestamps[frame_idx]
        elapsed = elapsed_time[frame_idx]

        fig.suptitle(
            f"ANYmal Joint States - {mission}\n"
            f"Frame: {frame_idx:06d}/{num_samples} | "
            f"Time: {elapsed:.2f}s | "
            f"Timestamp: {timestamp:.3f}s",
            fontsize=11,
        )

        return lines_pos + lines_vel + lines_torque

    # Set initial title
    update_frame(0)

    # Calculate interval
    interval_ms = int(1000 / fps)

    anim = FuncAnimation(
        fig, update_frame, frames=num_samples, interval=interval_ms, blit=False
    )

    plt.tight_layout()
    return fig, anim


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize ANYmal actuator data")
    parser.add_argument(
        "--mode",
        choices=["static", "movie"],
        default="static",
        help="Visualization mode (default: static)",
    )
    parser.add_argument(
        "--lite", action="store_true", help="Lite mode: only first 5000 samples"
    )
    return parser.parse_args()


def main() -> None:
    """Main function to visualize actuator data."""
    args = parse_args()
    mission = "2024-10-01-11-29-55"
    cache_dir = "./cache"
    max_frames = 5000 if args.lite else None

    print("=" * 60)
    mode_name = args.mode.upper()
    lite_str = " (Lite Mode)" if args.lite else ""
    print(f"ANYmal Actuator Visualization - {mode_name}{lite_str}")
    print("=" * 60)

    # Load actuator tar
    print(f"\nDownloading actuator data from mission: {mission}")
    tar_path = load_actuator_data(mission, cache_dir)
    print(f"Downloaded: {tar_path.name}")

    # Extract zarr data
    zarr_path = extract_actuator_zarr(tar_path, cache_dir)
    print(f"Extracted to: {zarr_path}")

    # Load joint data
    print("\nLoading joint data...")
    positions, velocities, torques, timestamps = load_joint_data(zarr_path, max_frames)
    print(f"Loaded {len(timestamps):,} samples for 12 joints")

    # Print statistics
    print_joint_stats(positions, velocities, torques, timestamps)

    # Visualize
    print(f"\nCreating {args.mode} visualization...")

    if args.mode == "static":
        visualize_joints_static(positions, velocities, torques, timestamps, mission)
    elif args.mode == "movie":
        # Calculate FPS
        if len(timestamps) > 1:
            dt = np.diff(timestamps[:100])
            fps = 1.0 / np.mean(dt)
        else:
            fps = 400.0
        print(f"Measured FPS: {fps:.2f}")
        create_joint_animation(positions, velocities, torques, timestamps, mission, fps)

    print("\nDisplaying visualization (close window to exit)...")
    plt.show()


if __name__ == "__main__":
    main()
