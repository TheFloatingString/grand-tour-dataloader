"""Visualize front camera data from Grand Tour dataset."""

import tarfile
from pathlib import Path

from grand_tour_dataloader import (
    GrandTourDataLoader,
    DataConfig,
    SensorType,
)


def extract_tar_file(tar_path: Path, extract_dir: Path) -> Path:
    """Extract tar file to specified directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(extract_dir)
    return extract_dir


def load_front_camera_data(
    mission: str, cache_dir: str
) -> dict[str, list[Path]]:
    """Load front camera data from specified mission."""
    config = DataConfig(
        include_lidar=False,
        include_cameras=True,
        camera_positions={"front"},
        include_imu=False,
        include_state=False,
        include_localization=False,
        include_transforms=False,
    )

    loader = GrandTourDataLoader(
        mission_dir=mission, config=config, cache_dir=cache_dir
    )

    print(f"Loading front camera data from mission: {mission}")
    data = loader.load()
    print(f"Loaded {len(data)} camera streams")
    return data


def display_camera_info(data: dict[str, list[Path]]) -> None:
    """Display information about loaded camera data."""
    for sensor_name in sorted(data.keys()):
        print(f"\n{sensor_name}:")
        for tar_path in data[sensor_name]:
            print(f"  File: {tar_path.name}")
            print(f"  Size: {tar_path.stat().st_size / 1e6:.2f} MB")
            print(f"  Path: {tar_path}")


def main() -> None:
    """Main function to visualize front camera data."""
    mission = "2024-10-01-11-29-55"
    cache_dir = "./cache"

    print("=" * 60)
    print("Front Camera Visualization")
    print("=" * 60)

    # Load camera data
    data = load_front_camera_data(mission, cache_dir)

    # Display information
    display_camera_info(data)

    # Extract first camera tar file as example
    if data:
        first_sensor = sorted(data.keys())[0]
        first_tar = data[first_sensor][0]

        print(f"\n{'=' * 60}")
        print(f"Extracting {first_sensor} data...")
        print(f"{'=' * 60}")

        extract_dir = Path(cache_dir) / "extracted" / first_sensor
        extracted = extract_tar_file(first_tar, extract_dir)

        print(f"\nExtracted to: {extracted}")
        print("\nContents:")
        for item in sorted(extracted.rglob("*"))[:10]:
            if item.is_file():
                print(f"  {item.relative_to(extracted)}")


if __name__ == "__main__":
    main()
