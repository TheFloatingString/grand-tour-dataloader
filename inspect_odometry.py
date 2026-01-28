"""Inspect odometry data structure."""

from pathlib import Path

import zarr


def inspect_odometry() -> None:
    """Inspect the zarr structure of odometry data."""
    zarr_path = Path("./cache/extracted/anymal_state_odometry/anymal_state_odometry")

    if not zarr_path.exists():
        print("Error: Odometry data not extracted yet.")
        return

    print("=" * 70)
    print("ANYmal Odometry Zarr Structure")
    print("=" * 70)

    # Open zarr group
    group = zarr.open_group(str(zarr_path), mode="r")

    print("\nZarr group contents:")
    print(f"  Keys: {list(group.keys())}")

    # Inspect each array
    for key in group.keys():
        arr = group[key]
        print(f"\n'{key}' array:")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")

        # Show sample data
        if len(arr.shape) == 1 and arr.shape[0] > 0:
            print(f"  Sample values: {arr[:5]}")
        elif len(arr.shape) == 2 and arr.shape[0] > 0:
            print(f"  First row: {arr[0]}")


if __name__ == "__main__":
    inspect_odometry()
