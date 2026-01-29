"""
Example script demonstrating depth estimation with Depth-Anything-V2
and novel view rendering with gsplat on Modal.
"""

import modal
import torch
import numpy as np
from PIL import Image
from typing import Tuple

# Define Modal image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "torchvision",
    "transformers",
    "pillow",
    "numpy",
    "huggingface_hub",
    "gsplat",
)

app = modal.App("render-gsplat")


def load_depth_anything_model(
    model_name: str = "depth_anything_v2_vits.pth",
) -> torch.nn.Module:
    """Load the Depth-Anything-V2 model."""
    from transformers import AutoModelForDepthEstimation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    model = model.to(device)
    model.eval()
    return model


def estimate_depth(
    image_path: str, model: torch.nn.Module
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate depth from RGB image."""
    from transformers import AutoImageProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth

    depth = (
        torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    rgb = np.array(image)
    return rgb, depth


def create_point_cloud(
    rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert RGB-D to 3D point cloud."""
    h, w = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0

    return torch.from_numpy(points).float(), torch.from_numpy(colors).float()


def render_simple_projection(
    points: torch.Tensor,
    colors: torch.Tensor,
    width: int,
    height: int,
    viewmat: torch.Tensor,
    K: torch.Tensor,
) -> np.ndarray:
    """Simple CPU-based point projection renderer."""
    # Transform points to camera space
    points_h = torch.cat([points, torch.ones(points.shape[0], 1)], dim=1)
    points_cam = (viewmat @ points_h.T).T[:, :3]

    # Project to image plane
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (points_cam[:, 0] / points_cam[:, 2]) * fx + cx
    y = (points_cam[:, 1] / points_cam[:, 2]) * fy + cy
    z = points_cam[:, 2]

    # Filter valid points
    valid = (z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, z = x[valid], y[valid], z[valid]
    colors_valid = colors[valid]

    # Create image with simple splatting
    image = np.zeros((height, width, 3), dtype=np.float32)
    depth_buffer = np.full((height, width), float("inf"))

    for i in range(len(x)):
        xi, yi = int(x[i]), int(y[i])
        if 0 <= xi < width and 0 <= yi < height:
            if z[i] < depth_buffer[yi, xi]:
                depth_buffer[yi, xi] = z[i]
                image[yi, xi] = colors_valid[i].numpy()

    return image


def render_with_gsplat(
    points: torch.Tensor,
    colors: torch.Tensor,
    width: int,
    height: int,
    viewmat: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """Render point cloud using gsplat (CUDA required)."""
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using simple CPU renderer")
        img = render_simple_projection(points, colors, width, height, viewmat, K)
        return torch.from_numpy(img)

    from gsplat import rasterization

    device = torch.device("cuda")
    points = points.to(device)
    colors = colors.to(device)
    viewmat = viewmat.to(device)
    K = K.to(device)

    # Initialize Gaussian parameters
    scales = torch.ones_like(points) * 0.01
    quats = torch.zeros(points.shape[0], 4, device=device)
    quats[:, 0] = 1.0
    opacities = torch.ones(points.shape[0], device=device) * 0.9

    # Rasterization returns (rendered, alphas, meta)
    rendered, alphas, meta = rasterization(
        means=points,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
    )

    # Return first image from batch
    return rendered[0]


def get_first_center_rgb_image(mission: str, cache_dir: str) -> str:
    """Get first frame from ANYmal center RGB camera."""
    import tarfile
    from pathlib import Path
    from huggingface_hub import snapshot_download

    allow_patterns = [f"{mission}/images/alphasense_front_center.tar"]

    cache_path = snapshot_download(
        repo_id="leggedrobotics/grand_tour_dataset",
        allow_patterns=allow_patterns,
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    tar_path = Path(cache_path) / mission / "images"
    tar_path = tar_path / "alphasense_front_center.tar"

    extract_dir = Path(cache_dir) / "images" / "front_center"

    # Check if already extracted (try all image extensions)
    existing_images = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        existing_images = sorted(extract_dir.rglob(ext))
        if existing_images:
            break

    if not existing_images:
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(extract_dir)

        # Re-check for images with all extensions
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            existing_images = sorted(extract_dir.rglob(ext))
            if existing_images:
                break

    if not existing_images:
        raise FileNotFoundError(f"No images found in {extract_dir}")

    return str(existing_images[0])


def create_comparison_image(original: np.ndarray, rendered: np.ndarray) -> Image.Image:
    """Create side-by-side comparison image."""
    h, w = original.shape[:2]

    # Create canvas for side-by-side display
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    comparison[:, :w] = original
    comparison[:, w:] = rendered

    return Image.fromarray(comparison)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
)
def render_on_modal() -> Tuple[bytes, bytes]:
    """Main execution function running on Modal with GPU."""
    import io

    mission = "2024-10-01-11-29-55"
    cache_dir = "/tmp/cache"

    print("Downloading first center RGB frame...")
    image_path = get_first_center_rgb_image(mission, cache_dir)
    print(f"Loaded image: {image_path}")

    print("Loading depth estimation model...")
    model = load_depth_anything_model()

    print("Estimating depth...")
    rgb, depth = estimate_depth(image_path, model)

    h, w = rgb.shape[:2]
    fx = fy = w  # Approximate focal length
    intrinsics = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])

    print("Creating point cloud...")
    points, colors = create_point_cloud(rgb, depth, intrinsics)

    # Define new camera pose (translate camera down)
    viewmat = torch.eye(4)
    viewmat[:3, 3] = torch.tensor([0.0, -0.3, 0.0])  # Move down

    K = torch.from_numpy(intrinsics).float()

    print("Rendering from new pose with gsplat...")
    rendered_image = render_with_gsplat(points, colors, w, h, viewmat, K)

    # Convert rendered to uint8
    output = (rendered_image.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

    # Save single rendered image
    rendered_buf = io.BytesIO()
    Image.fromarray(output).save(rendered_buf, format="PNG")

    # Create and save comparison image
    comparison_img = create_comparison_image(rgb, output)
    comparison_buf = io.BytesIO()
    comparison_img.save(comparison_buf, format="PNG")

    print("Rendering complete!")

    return rendered_buf.getvalue(), comparison_buf.getvalue()


@app.local_entrypoint()
def main() -> None:
    """Local entrypoint that calls Modal function."""
    print("Starting render on Modal with GPU...")
    rendered_result, comparison_result = render_on_modal.remote()

    # Save rendered image
    with open("rendered_output.png", "wb") as f:
        f.write(rendered_result)

    # Save comparison image
    with open("comparison_output.png", "wb") as f:
        f.write(comparison_result)

    print("Rendered image saved to rendered_output.png")
    print("Comparison image saved to comparison_output.png")
