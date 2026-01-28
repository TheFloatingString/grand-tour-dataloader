"""Core dataloader implementation for Grand Tour dataset."""

import tarfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from datasets import load_dataset


class SensorType(Enum):
    """Available sensor types in the Grand Tour dataset."""

    # LiDAR sensors
    HESAI_RAW = "hesai"
    HESAI_UNDISTORTED = "hesai_undistorted"
    HESAI_FILTERED = "hesai_filtered"
    LIVOX_RAW = "livox_points"
    LIVOX_UNDISTORTED = "livox_points_undistorted"
    LIVOX_FILTERED = "livox_points_filtered"
    DLIO_HESAI = "dlio_hesai_points_undistorted"

    # IMU sensors
    ADIS_IMU = "adis_imu"
    ALPHASENSE_IMU = "alphasense_imu"
    ANYMAL_IMU = "anymal_imu"
    STIM320_IMU = "stim320_imu"
    CPT7_IMU = "cpt7_imu"
    LIVOX_IMU = "livox_imu"

    # Cameras
    ALPHASENSE_FRONT_CENTER = "alphasense_front_center"
    ALPHASENSE_FRONT_LEFT = "alphasense_front_left"
    ALPHASENSE_FRONT_RIGHT = "alphasense_front_right"
    ALPHASENSE_LEFT = "alphasense_left"
    ALPHASENSE_RIGHT = "alphasense_right"
    HDR_FRONT = "hdr_front"
    HDR_LEFT = "hdr_left"
    HDR_RIGHT = "hdr_right"
    ZED2I_DEPTH = "zed2i_depth_image"
    ZED2I_DEPTH_CONFIDENCE = "zed2i_depth_confidence_image"
    ZED2I_LEFT = "zed2i_left_images"
    DEPTH_CAMERA_FRONT = "depth_camera_front"
    DEPTH_CAMERA_LEFT = "depth_camera_left"
    DEPTH_CAMERA_RIGHT = "depth_camera_right"

    # Robot state
    ANYMAL_ACTUATOR = "anymal_state_actuator"
    ANYMAL_BATTERY = "anymal_state_battery"
    ANYMAL_ODOMETRY = "anymal_state_odometry"
    ANYMAL_STATE_ESTIMATOR = "anymal_state_state_estimator"
    ANYMAL_COMMAND_TWIST = "anymal_command_twist"

    # Localization
    CPT7_ODOMETRY = "cpt7_ie_odometry"
    CPT7_TF = "cpt7_ie_tf"
    DLIO_ODOMETRY = "dlio_odometry"
    DLIO_TF = "dlio_tf"
    GNSS_RAW = "gnss_raw"

    # Transforms
    TF = "tf"


def _get_lidar_sensors(filtered_only: bool, undistorted_only: bool) -> Set[SensorType]:
    """Get LiDAR sensors based on filter options."""
    if filtered_only:
        return {SensorType.HESAI_FILTERED, SensorType.LIVOX_FILTERED}
    if undistorted_only:
        return {
            SensorType.HESAI_UNDISTORTED,
            SensorType.LIVOX_UNDISTORTED,
            SensorType.DLIO_HESAI,
        }
    return {
        SensorType.HESAI_RAW,
        SensorType.HESAI_UNDISTORTED,
        SensorType.HESAI_FILTERED,
        SensorType.LIVOX_RAW,
        SensorType.LIVOX_UNDISTORTED,
        SensorType.LIVOX_FILTERED,
        SensorType.DLIO_HESAI,
    }


def _get_imu_sensors() -> Set[SensorType]:
    """Get all IMU sensors."""
    return {
        SensorType.ADIS_IMU,
        SensorType.ALPHASENSE_IMU,
        SensorType.ANYMAL_IMU,
        SensorType.STIM320_IMU,
        SensorType.CPT7_IMU,
        SensorType.LIVOX_IMU,
    }


def _get_all_camera_sensors() -> List[SensorType]:
    """Get all camera sensors."""
    return [
        SensorType.ALPHASENSE_FRONT_CENTER,
        SensorType.ALPHASENSE_FRONT_LEFT,
        SensorType.ALPHASENSE_FRONT_RIGHT,
        SensorType.ALPHASENSE_LEFT,
        SensorType.ALPHASENSE_RIGHT,
        SensorType.HDR_FRONT,
        SensorType.HDR_LEFT,
        SensorType.HDR_RIGHT,
        SensorType.ZED2I_DEPTH,
        SensorType.ZED2I_DEPTH_CONFIDENCE,
        SensorType.ZED2I_LEFT,
        SensorType.DEPTH_CAMERA_FRONT,
        SensorType.DEPTH_CAMERA_LEFT,
        SensorType.DEPTH_CAMERA_RIGHT,
    ]


def _filter_cameras_by_position(
    cameras: List[SensorType], positions: Set[str]
) -> List[SensorType]:
    """Filter camera sensors by position keywords."""
    filtered = []
    for sensor in cameras:
        sensor_name = sensor.value.lower()
        if any(pos in sensor_name for pos in positions):
            filtered.append(sensor)
    return filtered


def _get_camera_sensors(
    positions: Optional[Set[str]],
) -> Set[SensorType]:
    """Get camera sensors based on position filter."""
    cameras = _get_all_camera_sensors()
    if positions:
        return set(_filter_cameras_by_position(cameras, positions))
    return set(cameras)


def _get_state_sensors() -> Set[SensorType]:
    """Get all robot state sensors."""
    return {
        SensorType.ANYMAL_ACTUATOR,
        SensorType.ANYMAL_BATTERY,
        SensorType.ANYMAL_ODOMETRY,
        SensorType.ANYMAL_STATE_ESTIMATOR,
        SensorType.ANYMAL_COMMAND_TWIST,
    }


def _get_localization_sensors() -> Set[SensorType]:
    """Get all localization sensors."""
    return {
        SensorType.CPT7_ODOMETRY,
        SensorType.CPT7_TF,
        SensorType.DLIO_ODOMETRY,
        SensorType.DLIO_TF,
        SensorType.GNSS_RAW,
    }


@dataclass
class DataConfig:
    """Configuration for filtering which data to load."""

    # High-level filters
    include_lidar: bool = True
    include_cameras: bool = True
    include_imu: bool = True
    include_state: bool = True
    include_localization: bool = True
    include_transforms: bool = True

    # Specific sensors to include (if None, all based on filters)
    specific_sensors: Optional[Set[SensorType]] = None

    # Specific sensors to exclude
    exclude_sensors: Set[SensorType] = field(default_factory=set)

    # LiDAR options
    lidar_filtered_only: bool = False
    lidar_undistorted_only: bool = False

    # Camera options
    camera_positions: Optional[Set[str]] = None

    def get_sensor_filters(self) -> Set[SensorType]:
        """Get final set of sensors to load based on config."""
        if self.specific_sensors is not None:
            return self.specific_sensors - self.exclude_sensors

        sensors = set()
        if self.include_lidar:
            sensors.update(
                _get_lidar_sensors(
                    self.lidar_filtered_only,
                    self.lidar_undistorted_only,
                )
            )
        if self.include_imu:
            sensors.update(_get_imu_sensors())
        if self.include_cameras:
            sensors.update(_get_camera_sensors(self.camera_positions))
        if self.include_state:
            sensors.update(_get_state_sensors())
        if self.include_localization:
            sensors.update(_get_localization_sensors())
        if self.include_transforms:
            sensors.add(SensorType.TF)

        return sensors - self.exclude_sensors


def _extract_mission_from_path(path: str) -> str:
    """Extract mission directory from file path."""
    if "/" in path:
        return path.split("/")[0]
    return ""


def _list_unique_missions(dataset: Any) -> List[str]:
    """Extract unique mission directories from dataset."""
    missions = set()
    for item in dataset:
        path = item.get("path", "")
        mission = _extract_mission_from_path(path)
        if mission:
            missions.add(mission)
    return sorted(list(missions))


def _matches_sensor(filename: str, sensor: SensorType) -> bool:
    """Check if filename matches sensor pattern."""
    sensor_name = sensor.value
    return filename.startswith(sensor_name) and filename.endswith(".tar")


def _collect_sensor_files(
    dataset: Any,
    mission_prefix: str,
    sensors: Set[SensorType],
) -> Dict[str, List[Dict[str, Any]]]:
    """Collect files for requested sensors from dataset."""
    sensor_files = {sensor.value: [] for sensor in sensors}

    for item in dataset:
        path = item.get("path", "")
        if not path.startswith(mission_prefix):
            continue

        filename = path.split("/")[-1]
        for sensor in sensors:
            if _matches_sensor(filename, sensor):
                sensor_files[sensor.value].append(item)
                break

    return sensor_files


def _filter_empty_sensors(
    sensor_files: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Remove sensors with no files."""
    return {name: items for name, items in sensor_files.items() if items}


class GrandTourDataLoader:
    """Dataloader for the Grand Tour legged robotics dataset.

    Example:
        >>> config = DataConfig(
        ...     include_lidar=True, include_cameras=False
        ... )
        >>> loader = GrandTourDataLoader(
        ...     mission_dir="2024-10-01-11-29-55",
        ...     config=config,
        ...     cache_dir="./cache"
        ... )
        >>> data = loader.load()
        >>> print(data.keys())
    """

    DATASET_NAME = "leggedrobotics/grand_tour_dataset"

    def __init__(
        self,
        mission_dir: Optional[str] = None,
        config: Optional[DataConfig] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        """Initialize the dataloader.

        Args:
            mission_dir: Mission directory name.
            config: Data configuration for filtering.
            cache_dir: Directory for caching downloaded data.
        """
        self.mission_dir = mission_dir
        self.config = config or DataConfig()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._dataset = None

    def list_missions(self) -> List[str]:
        """List all available mission directories.

        Returns:
            List of mission directory names.
        """
        ds = load_dataset(
            self.DATASET_NAME,
            split="train",
            streaming=True,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )
        return _list_unique_missions(ds)

    def load(self) -> Dict[str, Any]:
        """Load the dataset according to configuration.

        Returns:
            Dictionary mapping sensor names to their data.
        """
        if not self.mission_dir:
            raise ValueError("mission_dir must be specified to load data")

        sensors_to_load = self.config.get_sensor_filters()
        dataset = load_dataset(
            self.DATASET_NAME,
            split="train",
            streaming=True,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
        )

        mission_prefix = f"{self.mission_dir}/data/"
        sensor_files = _collect_sensor_files(dataset, mission_prefix, sensors_to_load)
        return _filter_empty_sensors(sensor_files)

    def extract_tar_contents(
        self, tar_item: Dict[str, Any], extract_path: Optional[Path] = None
    ) -> Path:
        """Extract a tar file from the dataset.

        Args:
            tar_item: Dataset item containing tar file data
            extract_path: Where to extract. If None, uses cache_dir.

        Returns:
            Path to extracted directory.
        """
        if extract_path is None:
            if self.cache_dir is None:
                raise ValueError("Either extract_path or cache_dir required")
            extract_path = self.cache_dir / "extracted" / tar_item["path"]

        extract_path = Path(extract_path)
        extract_path.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(fileobj=tar_item["content"]) as tar:
            tar.extractall(extract_path)

        return extract_path

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the mission.

        Returns:
            Dictionary containing metadata.
        """
        if not self.mission_dir:
            raise ValueError("mission_dir must be specified")

        sensors = self.config.get_sensor_filters()
        metadata = {
            "mission_dir": self.mission_dir,
            "dataset_name": self.DATASET_NAME,
            "config": {
                "include_lidar": self.config.include_lidar,
                "include_cameras": self.config.include_cameras,
                "include_imu": self.config.include_imu,
                "include_state": self.config.include_state,
                "include_localization": self.config.include_localization,
                "include_transforms": self.config.include_transforms,
            },
            "sensors": [s.value for s in sensors],
        }

        return metadata
