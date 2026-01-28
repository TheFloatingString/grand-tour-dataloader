"""Basic usage examples for the Grand Tour dataloader."""

from grand_tour_dataloader import (
    GrandTourDataLoader,
    DataConfig,
    SensorType,
)


def _print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _print_sensor_list(data: dict, description: str) -> None:
    """Print loaded sensor data."""
    print(f"\nLoaded {len(data)} {description}:")
    for sensor_name in sorted(data.keys()):
        print(f"  - {sensor_name}")


def _print_sensor_files(data: dict, description: str) -> None:
    """Print loaded sensor data with file counts."""
    print(f"\nLoaded {len(data)} {description}:")
    for sensor_name in sorted(data.keys()):
        num_files = len(data[sensor_name])
        print(f"  - {sensor_name}: {num_files} file(s)")


def _print_metadata_config(config: dict) -> None:
    """Print metadata configuration."""
    print("\n  Configuration:")
    for key, value in config.items():
        print(f"    {key}: {value}")


def _print_metadata_sensors(sensors: list) -> None:
    """Print metadata sensors."""
    print(f"\n  Sensors ({len(sensors)}):")
    for sensor in sensors:
        print(f"    - {sensor}")


def example_list_missions() -> None:
    """Example: List all available missions."""
    _print_section_header("Example 1: Listing all available missions")

    loader = GrandTourDataLoader()
    missions = loader.list_missions()

    print(f"\nFound {len(missions)} missions:")
    for mission in missions:
        print(f"  - {mission}")


def example_load_all_data() -> None:
    """Example: Load all data from a specific mission."""
    _print_section_header("Example 2: Loading all data from a mission")

    mission = "2024-10-01-11-29-55"
    config = DataConfig()
    loader = GrandTourDataLoader(
        mission_dir=mission, config=config, cache_dir="./cache"
    )

    print(f"\nLoading mission: {mission}")
    data = loader.load()
    _print_sensor_files(data, "sensor data streams")


def example_load_lidar_only() -> None:
    """Example: Load only LiDAR data."""
    _print_section_header("Example 3: Loading only LiDAR data")

    mission = "2024-10-01-11-29-55"
    config = DataConfig(
        include_lidar=True,
        include_cameras=False,
        include_imu=False,
        include_state=False,
        include_localization=False,
        include_transforms=False,
    )

    loader = GrandTourDataLoader(
        mission_dir=mission, config=config, cache_dir="./cache"
    )

    print(f"\nLoading LiDAR data from mission: {mission}")
    data = loader.load()
    _print_sensor_list(data, "LiDAR streams")


def example_load_filtered_lidar() -> None:
    """Example: Load only filtered LiDAR data."""
    _print_section_header("Example 4: Loading only filtered LiDAR")

    mission = "2024-10-01-11-29-55"
    config = DataConfig(
        include_lidar=True,
        lidar_filtered_only=True,
        include_cameras=False,
        include_imu=False,
        include_state=False,
        include_localization=False,
        include_transforms=False,
    )

    loader = GrandTourDataLoader(
        mission_dir=mission, config=config, cache_dir="./cache"
    )

    print(f"\nLoading filtered LiDAR data from mission: {mission}")
    data = loader.load()
    _print_sensor_list(data, "filtered LiDAR streams")


def example_load_front_cameras() -> None:
    """Example: Load only front-facing cameras."""
    _print_section_header("Example 5: Loading only front-facing cameras")

    mission = "2024-10-01-11-29-55"
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
        mission_dir=mission, config=config, cache_dir="./cache"
    )

    print(f"\nLoading front camera data from mission: {mission}")
    data = loader.load()
    _print_sensor_list(data, "front camera streams")


def example_load_specific_sensors() -> None:
    """Example: Load specific sensors using SensorType enum."""
    _print_section_header("Example 6: Loading specific sensors")

    mission = "2024-10-01-11-29-55"
    config = DataConfig(
        specific_sensors={
            SensorType.HESAI_UNDISTORTED,
            SensorType.ANYMAL_IMU,
            SensorType.ALPHASENSE_FRONT_CENTER,
            SensorType.ANYMAL_ODOMETRY,
        }
    )

    loader = GrandTourDataLoader(
        mission_dir=mission, config=config, cache_dir="./cache"
    )

    print(f"\nLoading specific sensors from mission: {mission}")
    data = loader.load()
    _print_sensor_list(data, "sensor streams")


def example_exclude_sensors() -> None:
    """Example: Load all data except specific sensors."""
    _print_section_header("Example 7: Loading data with exclusions")

    mission = "2024-10-01-11-29-55"
    config = DataConfig(
        include_lidar=True,
        include_cameras=True,
        include_imu=True,
        exclude_sensors={
            SensorType.HESAI_RAW,
            SensorType.ZED2I_DEPTH,
            SensorType.ZED2I_DEPTH_CONFIDENCE,
        },
    )

    loader = GrandTourDataLoader(
        mission_dir=mission, config=config, cache_dir="./cache"
    )

    print(f"\nLoading data with exclusions from mission: {mission}")
    data = loader.load()
    _print_sensor_list(data, "sensor streams (with exclusions)")


def example_get_metadata() -> None:
    """Example: Get metadata about the loaded configuration."""
    _print_section_header("Example 8: Getting metadata")

    mission = "2024-10-01-11-29-55"
    config = DataConfig(
        include_lidar=True,
        lidar_filtered_only=True,
        include_cameras=False,
    )

    loader = GrandTourDataLoader(
        mission_dir=mission, config=config, cache_dir="./cache"
    )

    metadata = loader.get_metadata()

    print(f"\nMetadata for mission: {mission}")
    print(f"  Dataset: {metadata['dataset_name']}")
    print(f"  Mission: {metadata['mission_dir']}")
    _print_metadata_config(metadata["config"])
    _print_metadata_sensors(metadata["sensors"])


def _print_intro() -> None:
    """Print introductory message."""
    print("Grand Tour Dataloader - Usage Examples")
    print("=" * 60)
    print("\nNote: Examples will download data from HuggingFace.")
    print("Ensure internet connection and sufficient storage.\n")


def _print_completion() -> None:
    """Print completion message."""
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run examples
    # Note: Some examples may take time to download data

    _print_intro()

    # Example 1: List missions
    # example_list_missions()

    # Example 2: Load all data
    # example_load_all_data()

    # Example 3: Load LiDAR only
    # example_load_lidar_only()

    # Example 4: Load filtered LiDAR
    # example_load_filtered_lidar()

    # Example 5: Load front cameras
    example_load_front_cameras()

    # Example 6: Load specific sensors
    # example_load_specific_sensors()

    # Example 7: Load with exclusions
    # example_exclude_sensors()

    # Example 8: Get metadata
    # example_get_metadata()

    _print_completion()
