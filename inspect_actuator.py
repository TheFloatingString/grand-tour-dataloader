"""Inspect actuator data structure."""

import tarfile
from pathlib import Path

import zarr


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


def inspect_actuator() -> None:
    """Inspect the zarr structure of actuator data."""
    mission = "2024-10-01-11-29-55"
    cache_dir = "./cache"

    print("=" * 70)
    print("ANYmal Actuator Data Inspection")
    print("=" * 70)

    # Load and extract
    print(f"\nDownloading actuator data from mission: {mission}")
    tar_path = load_actuator_data(mission, cache_dir)
    print(f"Downloaded: {tar_path.name}")

    zarr_path = extract_actuator_zarr(tar_path, cache_dir)
    print(f"Extracted to: {zarr_path}")

    # Open zarr group
    group = zarr.open_group(str(zarr_path), mode="r")

    print("\nZarr group contents:")
    print(f"  Keys: {sorted(list(group.keys()))}")

    # Inspect each array
    for key in sorted(group.keys()):
        arr = group[key]
        print(f"\n'{key}' array:")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")

        # Show sample data
        if len(arr.shape) == 1 and arr.shape[0] > 0:
            print(f"  Sample values (first 3): {arr[:3]}")
        elif len(arr.shape) == 2 and arr.shape[0] > 0:
            print(f"  First row: {arr[0]}")
            print(f"  Num joints: {arr.shape[1]}")


if __name__ == "__main__":
    inspect_actuator()
