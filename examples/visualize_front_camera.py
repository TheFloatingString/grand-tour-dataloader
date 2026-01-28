"""Visualize front camera data as 3-view movie using matplotlib."""

import argparse
import tarfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image


def extract_tar_file(tar_path: Path, extract_dir: Path) -> Path:
    """Extract tar file to specified directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(extract_dir, filter="data")
    return extract_dir


def load_front_camera_images(mission: str, cache_dir: str) -> dict[str, list[Path]]:
    """Load front camera image tars from mission."""
    from huggingface_hub import snapshot_download

    allow_patterns = [
        f"{mission}/images/alphasense_front_left.tar",
        f"{mission}/images/alphasense_front_center.tar",
        f"{mission}/images/alphasense_front_right.tar",
    ]

    cache_path = snapshot_download(
        repo_id="leggedrobotics/grand_tour_dataset",
        allow_patterns=allow_patterns,
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    images_path = Path(cache_path) / mission / "images"
    camera_files = {}

    for tar_file in images_path.glob("*.tar"):
        sensor_name = tar_file.stem
        camera_files[sensor_name] = [tar_file]

    return camera_files


def extract_images_from_tar(
    tar_path: Path, cache_dir: str, max_frames: int | None = None
) -> list[Path]:
    """Extract images from tar and return sorted paths."""
    sensor_name = tar_path.stem
    extract_dir = Path(cache_dir) / "images" / sensor_name

    # Only extract if not already done
    if not extract_dir.exists():
        extract_tar_file(tar_path, extract_dir)

    # Get all image files sorted
    image_files = list(extract_dir.rglob("*.png"))
    image_files.extend(extract_dir.rglob("*.jpg"))
    image_files.extend(extract_dir.rglob("*.jpeg"))
    all_images = sorted(image_files)

    if max_frames is not None:
        return all_images[:max_frames]
    return all_images


def load_image_as_array(image_path: Path) -> np.ndarray:
    """Load image file as numpy array."""
    img = Image.open(image_path)
    return np.array(img)


def create_three_view_animation(
    image_lists: list[list[Path]],
    titles: list[str],
    fps: float = 10.0,
) -> tuple[plt.Figure, FuncAnimation]:
    """Create 3-view matplotlib animation from image lists."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Front Cameras - 3 View", fontsize=16)

    # Load first frame from each camera
    images = []
    for ax, img_list, title in zip(axes, image_lists, titles):
        frame = load_image_as_array(img_list[0])
        if frame.ndim == 2:
            im = ax.imshow(frame, cmap="gray")
        else:
            im = ax.imshow(frame)
        ax.set_title(title)
        ax.axis("off")
        images.append(im)

    # Get minimum frame count
    min_frames = min(len(imgs) for imgs in image_lists)

    def update_frame(frame_idx: int) -> list[Any]:
        """Update function for animation."""
        for im, img_list in zip(images, image_lists):
            frame = load_image_as_array(img_list[frame_idx])
            im.set_array(frame)

        # Calculate time in seconds
        time_sec = frame_idx / fps
        fig.suptitle(
            f"Frame: {frame_idx:04d}/{min_frames} | Time: {time_sec:.2f}s",
            fontsize=16,
        )
        return images

    # Real-time playback: interval in milliseconds
    interval_ms = int(1000 / fps)

    anim = FuncAnimation(
        fig,
        update_frame,
        frames=min_frames,
        interval=interval_ms,
        blit=True,
    )

    plt.tight_layout()
    return fig, anim


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize front camera data as 3-view movie"
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Lite mode: only display first 2400 frames",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to visualize front camera data."""
    args = parse_args()
    mission = "2024-10-01-11-29-55"
    cache_dir = "./cache"
    max_frames = 2400 if args.lite else None

    print("=" * 60)
    mode = "Lite Mode (2400 frames)" if args.lite else "Full Mode"
    print(f"Front Camera 3-View Visualization - {mode}")
    print("=" * 60)

    # Load camera image tars
    print(f"\nDownloading camera images from mission: {mission}")
    data = load_front_camera_images(mission, cache_dir)
    print(f"Downloaded {len(data)} camera image tars")

    # Select 3 cameras
    camera_names = [
        "alphasense_front_left",
        "alphasense_front_center",
        "alphasense_front_right",
    ]

    print("\nExtracting images from tars...")
    image_lists = []
    titles = []

    for name in camera_names:
        if name in data and data[name]:
            tar_path = data[name][0]
            print(f"  Extracting {name}...")
            images = extract_images_from_tar(tar_path, cache_dir, max_frames)
            print(f"    Found {len(images)} images")
            image_lists.append(images)
            titles.append(name.replace("alphasense_", ""))

    if len(image_lists) < 3:
        print("\nError: Need at least 3 cameras")
        return

    # Camera runs at 10 FPS (real-time playback)
    fps = 10.0
    total_time = len(image_lists[0]) / fps

    print("\nCreating animation:")
    print(f"  Frames: {len(image_lists[0])}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {total_time:.1f}s")

    fig, anim = create_three_view_animation(image_lists, titles, fps)

    print("\nDisplaying 3-view camera movie (real-time)...")
    print("Close the window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
