"""
AI Fire Detection System - Automated Model Training
=====================================================
Generates synthetic fire training data and trains YOLOv8 model.
No external downloads or API keys required.

Usage:
    python train_model.py                      # Quick training (15 epochs)
    python train_model.py --epochs 50          # Full training
    python train_model.py --api-key KEY        # Use Roboflow dataset (best accuracy)
"""

import os
import sys
import shutil
import random
import argparse
import logging
import time
import math

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "fire_smoke")

os.makedirs(MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════
#  SYNTHETIC FIRE DATASET GENERATOR
# ═══════════════════════════════════════════════════════

def draw_fire_shape(img, cx, cy, max_radius):
    """Draw a realistic fire-like shape using layered ellipses and gradients."""
    h, w = img.shape[:2]
    overlay = img.copy()

    num_layers = random.randint(8, 20)
    for i in range(num_layers):
        # Fire colors: white core -> yellow -> orange -> red
        t = i / num_layers
        if t < 0.2:
            color = (random.randint(200, 255), random.randint(220, 255), random.randint(240, 255))  # BGR white-yellow
        elif t < 0.5:
            color = (random.randint(30, 100), random.randint(180, 255), random.randint(220, 255))   # BGR yellow-orange
        elif t < 0.8:
            color = (random.randint(0, 50), random.randint(80, 180), random.randint(200, 255))      # BGR orange-red
        else:
            color = (random.randint(0, 30), random.randint(0, 80), random.randint(150, 220))        # BGR dark red

        rx = int(max_radius * (0.3 + 0.7 * t) * random.uniform(0.5, 1.2))
        ry = int(max_radius * (0.4 + 0.8 * t) * random.uniform(0.6, 1.4))

        offset_x = int(random.gauss(0, max_radius * 0.15))
        offset_y = int(random.gauss(0, max_radius * 0.2)) - int(t * max_radius * 0.3)
        angle = random.uniform(-20, 20)

        center = (max(0, min(w-1, cx + offset_x)), max(0, min(h-1, cy + offset_y)))
        axes = (max(1, rx), max(1, ry))

        cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1)

    # Blend
    alpha = random.uniform(0.5, 0.85)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Add glow effect
    glow = np.zeros_like(img)
    glow_radius = int(max_radius * 1.5)
    cv2.circle(glow, (cx, cy), glow_radius, (30, 100, 180), -1)
    glow = cv2.GaussianBlur(glow, (0, 0), glow_radius // 2)
    cv2.addWeighted(img, 1, glow, 0.3, 0, img)

    return img


def generate_background(w, h, bg_type=None):
    """Generate varied background images."""
    if bg_type is None:
        bg_type = random.choice(["dark", "room", "outdoor", "night", "gray", "texture"])

    img = np.zeros((h, w, 3), dtype=np.uint8)

    if bg_type == "dark":
        val = random.randint(5, 40)
        img[:] = (val, val, val)
        # Add noise
        noise = np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
        img = cv2.add(img, noise)

    elif bg_type == "room":
        # Indoor scene - walls and floor
        wall_color = (random.randint(140, 200), random.randint(140, 190), random.randint(140, 180))
        floor_color = (random.randint(80, 130), random.randint(70, 120), random.randint(60, 100))
        horizon = random.randint(h//3, 2*h//3)
        img[:horizon] = wall_color
        img[horizon:] = floor_color
        noise = np.random.randint(0, 15, (h, w, 3), dtype=np.uint8)
        img = cv2.add(img, noise)

    elif bg_type == "outdoor":
        # Sky and ground
        sky = (random.randint(150, 220), random.randint(120, 180), random.randint(80, 140))
        ground = (random.randint(50, 100), random.randint(80, 140), random.randint(60, 120))
        horizon = random.randint(h//4, h//2)
        for y in range(h):
            t = y / h
            if y < horizon:
                img[y] = sky
            else:
                img[y] = ground
        noise = np.random.randint(0, 10, (h, w, 3), dtype=np.uint8)
        img = cv2.add(img, noise)

    elif bg_type == "night":
        img[:] = (random.randint(10, 30), random.randint(10, 25), random.randint(10, 20))
        noise = np.random.randint(0, 8, (h, w, 3), dtype=np.uint8)
        img = cv2.add(img, noise)

    elif bg_type == "gray":
        val = random.randint(80, 180)
        img[:] = (val, val, val)
        noise = np.random.randint(0, 25, (h, w, 3), dtype=np.uint8)
        img = cv2.add(img, noise)

    elif bg_type == "texture":
        # Random colored texture
        base = np.random.randint(40, 200, (h//8, w//8, 3), dtype=np.uint8)
        img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)

    return img


def generate_fire_image(w=640, h=640):
    """Generate a synthetic image with fire and return the bounding box."""
    img = generate_background(w, h)

    # Random fire position and size
    fire_w = random.randint(w // 6, w // 2)
    fire_h = random.randint(h // 6, h // 2)
    cx = random.randint(fire_w // 2 + 10, w - fire_w // 2 - 10)
    cy = random.randint(fire_h // 2 + 10, h - fire_h // 2 - 10)

    max_radius = max(fire_w, fire_h) // 2

    # Draw fire
    img = draw_fire_shape(img, cx, cy, max_radius)

    # Add some random brightness/contrast variation
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Calculate YOLO-format bounding box (normalized)
    x1 = max(0, cx - fire_w // 2) / w
    y1 = max(0, cy - fire_h // 2) / h
    x2 = min(w, cx + fire_w // 2) / w
    y2 = min(h, cy + fire_h // 2) / h
    bbox_cx = (x1 + x2) / 2
    bbox_cy = (y1 + y2) / 2
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # class 0 = fire
    label = f"0 {bbox_cx:.6f} {bbox_cy:.6f} {bbox_w:.6f} {bbox_h:.6f}"

    return img, label


def generate_smoke_image(w=640, h=640):
    """Generate a synthetic image with smoke."""
    img = generate_background(w, h)

    # Smoke: gray/white translucent blobs
    num_blobs = random.randint(3, 8)
    smoke_cx = random.randint(w // 4, 3 * w // 4)
    smoke_cy = random.randint(h // 4, 3 * h // 4)
    smoke_radius = random.randint(w // 6, w // 3)

    overlay = img.copy()
    for _ in range(num_blobs):
        bx = smoke_cx + random.randint(-smoke_radius, smoke_radius)
        by = smoke_cy + random.randint(-smoke_radius, smoke_radius)
        br = random.randint(smoke_radius // 3, smoke_radius)
        gray_val = random.randint(150, 230)
        color = (gray_val, gray_val, gray_val + random.randint(-10, 10))
        cv2.circle(overlay, (bx, by), br, color, -1)

    # Heavy blur for smoke effect
    overlay = cv2.GaussianBlur(overlay, (0, 0), smoke_radius // 2)
    alpha = random.uniform(0.3, 0.6)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Bounding box
    x1 = max(0, smoke_cx - smoke_radius) / w
    y1 = max(0, smoke_cy - smoke_radius) / h
    x2 = min(w, smoke_cx + smoke_radius) / w
    y2 = min(h, smoke_cy + smoke_radius) / h
    bbox_cx = (x1 + x2) / 2
    bbox_cy = (y1 + y2) / 2
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # class 1 = smoke
    label = f"1 {bbox_cx:.6f} {bbox_cy:.6f} {bbox_w:.6f} {bbox_h:.6f}"

    return img, label


def generate_negative_image(w=640, h=640):
    """Generate an image WITHOUT fire/smoke (negative sample)."""
    img = generate_background(w, h, random.choice(["room", "outdoor", "gray", "texture"]))

    # Add some random colored objects that are NOT fire
    for _ in range(random.randint(0, 5)):
        x = random.randint(0, w)
        y = random.randint(0, h)
        r = random.randint(20, 100)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        shape = random.choice(["circle", "rect"])
        if shape == "circle":
            cv2.circle(img, (x, y), r, color, -1)
        else:
            cv2.rectangle(img, (x - r, y - r), (x + r, y + r), color, -1)

    return img, ""  # No label (negative)


def generate_dataset(num_train=300, num_val=60):
    """Generate complete synthetic fire detection dataset."""
    logger.info(f"Generating synthetic dataset: {num_train} train + {num_val} val images...")

    for split, count in [("train", num_train), ("valid", num_val)]:
        img_dir = os.path.join(DATASET_DIR, split, "images")
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(count):
            # 50% fire, 20% smoke, 30% negative
            r = random.random()
            if r < 0.5:
                img, label = generate_fire_image()
                prefix = "fire"
            elif r < 0.7:
                img, label = generate_smoke_image()
                prefix = "smoke"
            else:
                img, label = generate_negative_image()
                prefix = "neg"

            fname = f"{prefix}_{split}_{i:04d}"
            cv2.imwrite(os.path.join(img_dir, f"{fname}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            lbl_path = os.path.join(lbl_dir, f"{fname}.txt")
            with open(lbl_path, "w") as f:
                if label:
                    f.write(label + "\n")

            if (i + 1) % 50 == 0:
                logger.info(f"  [{split}] Generated {i+1}/{count} images")

    # Create data.yaml
    data_yaml = os.path.join(DATASET_DIR, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"""path: {DATASET_DIR.replace(chr(92), '/')}
train: train/images
val: valid/images

nc: 2
names:
  0: fire
  1: smoke
""")

    logger.info(f"Dataset generated: {num_train + num_val} total images")
    logger.info(f"  Config: {data_yaml}")
    return data_yaml


# ═══════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════

def train_model(data_yaml, epochs=15, model_name="yolov8n.pt", batch_size=8, img_size=640):
    """Train YOLOv8 model on fire detection dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print()
    logger.info("=" * 60)
    logger.info("FIRE DETECTION MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"  Base model:    {model_name}")
    logger.info(f"  Dataset:       {data_yaml}")
    logger.info(f"  Epochs:        {epochs}")
    logger.info(f"  Batch size:    {batch_size}")
    logger.info(f"  Image size:    {img_size}")
    logger.info("=" * 60)
    print()

    logger.info("Loading YOLOv8 base model...")
    model = YOLO(model_name)

    logger.info("Starting training...")
    start_time = time.time()

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name="fire_detection",
        project=os.path.join(BASE_DIR, "runs"),
        exist_ok=True,
        patience=max(10, epochs // 3),
        save=True,
        plots=True,
        verbose=True,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.15,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
    )

    elapsed = time.time() - start_time

    # Copy best model
    best_src = os.path.join(BASE_DIR, "runs", "fire_detection", "weights", "best.pt")
    last_src = os.path.join(BASE_DIR, "runs", "fire_detection", "weights", "last.pt")
    best_dst = os.path.join(MODEL_DIR, "best.pt")

    source = best_src if os.path.exists(best_src) else last_src

    if os.path.exists(source):
        shutil.copy2(source, best_dst)
        print()
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"  Model saved: {best_dst}")
        logger.info(f"  Time:        {elapsed/60:.1f} minutes")
        logger.info("")
        logger.info("  Restart the app to use the trained model:")
        logger.info("    python app.py")
        logger.info("=" * 60)
    else:
        logger.warning("Training done but weights not found at expected path.")
        # Search for weights
        for root, dirs, files in os.walk(os.path.join(BASE_DIR, "runs")):
            for f in files:
                if f in ("best.pt", "last.pt"):
                    src = os.path.join(root, f)
                    shutil.copy2(src, best_dst)
                    logger.info(f"  Found and copied model from: {src}")
                    break

    return results


def setup_roboflow_dataset(api_key):
    """Download dataset from Roboflow (for best accuracy)."""
    try:
        from roboflow import Roboflow
        logger.info("Connecting to Roboflow...")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project("fire-detection-pbnme")
        version = project.version(1)
        dataset = version.download("yolov8", location=DATASET_DIR)
        data_yaml = os.path.join(DATASET_DIR, "data.yaml")
        return data_yaml if os.path.exists(data_yaml) else None
    except Exception as e:
        logger.error(f"Roboflow download failed: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Fire Detection Model")
    parser.add_argument("--api-key", type=str, help="Roboflow API key (for real dataset)")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs (default: 15)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--data", type=str, help="Path to existing data.yaml")
    parser.add_argument("--num-images", type=int, default=300, help="Synthetic train images to generate")

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("   FIREGUARD AI - MODEL TRAINING PIPELINE")
    print("=" * 60)
    print()

    # Step 1: Prepare dataset
    if args.data:
        data_yaml = args.data
    elif args.api_key:
        data_yaml = setup_roboflow_dataset(args.api_key)
    else:
        data_yaml = generate_dataset(num_train=args.num_images, num_val=max(30, args.num_images // 5))

    if data_yaml is None or not os.path.exists(data_yaml):
        logger.error("Failed to prepare dataset.")
        sys.exit(1)

    # Step 2: Train
    train_model(
        data_yaml=data_yaml,
        epochs=args.epochs,
        model_name=args.model,
        batch_size=args.batch,
        img_size=args.img_size,
    )
