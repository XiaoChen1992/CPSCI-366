# -*- coding: utf-8 -*-
"""
Download Oxford-IIIT Pet (via torchvision) and visualize images + segmentation masks.
- Supports trimap (3-class) or binary (merge border into pet).
- Saves a visualization grid to ./samples/oxpet_viz_grid.png
- Also prints basic dataset stats.
Usage:
    python oxpet_download_and_viz.py --root ~/data --split trainval --classes trimap --n 12
"""
import argparse
from pathlib import Path
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms as T

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/Users/schen3/Downloads/data",
                   help="Data root directory (should contain oxford-iiit-pet/ subdirectory)")
    p.add_argument("--split", type=str, default="trainval",
                   choices=["trainval", "test"], help="Data split")
    p.add_argument("--classes", type=str, default="trimap",
                   choices=["trimap", "binary"],
                   help="trimap=3 classes (pet/background/border); binary=2 classes (border merged into foreground)")
    p.add_argument("--resize", type=int, default=512,
                   help="Resize to this size on the short side (images and masks resized together)")
    p.add_argument("--n", type=int, default=12, help="Number of visualization samples")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save-dir", type=str, default="samples", help="Output directory")
    return p.parse_args()

def mask_to_classes(mask_pil: Image.Image, mode: str = "trimap"):
    """
    Official trimap values: 1=pet/foreground, 2=background, 3=border
    Returns: integer numpy array (H,W) and list of class names
    """
    m = np.array(mask_pil, dtype=np.int64)
    if mode == "trimap":
        # Map to {0,1,2} for easier coloring and computation: 0=pet, 1=background, 2=border
        m = m - 1
        class_names = ["pet", "background", "border"]
    else:
        # binary: merge border into foreground
        pet = (m == 1) | (m == 3)
        m = pet.astype(np.int64)  # 1=pet, 0=background
        class_names = ["background", "pet"]
    return m, class_names

def colorize_mask(mask: np.ndarray, class_names):
    """Map integer label mask to RGB for visualization (manual colors for classes)."""
    H, W = mask.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    if len(class_names) == 3:
        # 0=pet(red), 1=background(black), 2=border(green)
        vis[mask == 0] = (220, 20, 60)   # red
        vis[mask == 1] = (0, 0, 0)       # black
        vis[mask == 2] = (34, 139, 34)   # green
    else:
        # Binary: 1=pet(red), 0=background(black)
        vis[mask == 1] = (220, 20, 60)
        vis[mask == 0] = (0, 0, 0)
    return Image.fromarray(vis, mode="RGB")

def overlay(image_pil: Image.Image, mask_rgb: Image.Image, alpha=0.45):
    """Overlay using PIL's blend to avoid dtype/size issues."""
    if image_pil.size != mask_rgb.size:
        mask_rgb = mask_rgb.resize(image_pil.size, resample=Image.NEAREST)
    img_rgb = image_pil.convert("RGB")
    return Image.blend(img_rgb, mask_rgb, alpha)

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.root).expanduser()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resize images and masks the same way (use nearest for masks to avoid interpolation artifacts)
    img_tf = T.Resize(args.resize, antialias=True)
    mask_tf = T.Resize(args.resize, interpolation=Image.NEAREST)

    # Do not download: download=False. If files are missing locally, an exception will be raised with guidance.
    try:
        ds = OxfordIIITPet(
            root=str(root),
            split=args.split,
            target_types="segmentation",
            transform=img_tf,
            target_transform=mask_tf,
            download=False
        )
    except Exception as e:
        print("\n[Error] Failed to load data:", e)
        print("Please confirm the directory exists:")
        print(f"  {root}/oxford-iiit-pet/images/")
        print(f"  {root}/oxford-iiit-pet/annotations/trimaps/")
        print("And ensure files are extracted (official annotations.tar.gz contains trimaps/ and xmls/).")
        return

    print(f"Loaded Oxford-IIIT Pet: split={args.split}, size={len(ds)}")

    n = min(args.n, len(ds))
    idxs = random.sample(range(len(ds)), n)

    # Create grid: each row shows (image, mask, overlay)
    cols = 3
    rows = n
    fig_w = 10
    fig_h = max(6, rows * 2.2)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    for r, idx in enumerate(idxs):
        img_pil, mask_pil = ds[idx]  # both already resized
        mask_arr, class_names = mask_to_classes(mask_pil, mode=args.classes)
        mask_rgb = colorize_mask(mask_arr, class_names)
        over = overlay(img_pil, mask_rgb, alpha=0.45)

        if rows == 1:
            ax_img, ax_mask, ax_over = axes[0], axes[1], axes[2]
        else:
            ax_img, ax_mask, ax_over = axes[r]

        ax_img.imshow(img_pil)
        ax_img.set_title("Image")
        ax_img.axis("off")

        ax_mask.imshow(mask_rgb)
        ax_mask.set_title(f"Mask ({args.classes})")
        ax_mask.axis("off")

        ax_over.imshow(over)
        ax_over.set_title("Overlay")
        ax_over.axis("off")

        # Also save the first few individual overlay images
        if r < 4:
            over.save(save_dir / f"overlay_{args.classes}_{r}.png")

    plt.tight_layout()
    out_path = save_dir / "oxpet_viz_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved visualization grid to: {out_path.resolve()}")
    print("Class names:", class_names)

if __name__ == "__main__":
    main()
