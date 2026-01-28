"""Grand Tour Dataset Dataloader.

A dataloader for the Grand Tour legged robotics dataset from ETH Zürich.
https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset
"""

from .dataloader import GrandTourDataLoader, DataConfig, SensorType

__version__ = "0.1.0"
__all__ = ["GrandTourDataLoader", "DataConfig", "SensorType"]


def _print_missions(missions: list[str]) -> None:
    """Print list of available missions."""
    print(f"\nAvailable missions ({len(missions)}):")
    for mission in missions:
        print(f"  - {mission}")


def _print_data_streams(data: dict) -> None:
    """Print loaded data streams."""
    print(f"Loaded {len(data)} data streams")
    for key in data.keys():
        print(f"  - {key}")


def _handle_list_missions() -> None:
    """Handle the list-missions command."""
    loader = GrandTourDataLoader()
    print("Fetching available missions...")
    missions = loader.list_missions()
    _print_missions(missions)


def _handle_load_mission(mission: str, cache_dir: str) -> None:
    """Handle loading a specific mission."""
    config = DataConfig(
        include_lidar=True,
        include_cameras=True,
        include_imu=True,
        include_state=True,
    )
    loader = GrandTourDataLoader(
        mission_dir=mission, config=config, cache_dir=cache_dir
    )
    print(f"Loading mission: {mission}")
    data = loader.load()
    _print_data_streams(data)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Grand Tour Dataset Dataloader")
    parser.add_argument(
        "--mission",
        type=str,
        help="Mission directory name (e.g., 2024-10-01-11-29-55)",
    )
    parser.add_argument(
        "--list-missions",
        action="store_true",
        help="List available missions",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Cache directory",
    )

    args = parser.parse_args()

    if args.list_missions:
        _handle_list_missions()
    elif args.mission:
        _handle_load_mission(args.mission, args.cache_dir)
    else:
        parser.print_help()
