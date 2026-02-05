import os
import json
import numpy as np
import cv2
import tifffile as tiff
from collections import defaultdict
from pycocotools import mask as coco_mask
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="COCO processing for building footprint datasets"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["semantic", "instance", "negatives", "stats"],
        help="Processing mode"
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Dataset split"
    )

    parser.add_argument(
        "--common_path",
        type=str,
        required=True,
        help="Root dataset path"
    )

    parser.add_argument(
        "--pad_ratio",
        type=float,
        default=0.1,
        help="Padding ratio for instance crops"
    )

    return parser.parse_args()

    
#local path on macbook
#common_path = '/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/WHUBuildingDataset'

#path of the nas
common_path = '/mnt/nas151/sar/Footprint/datasets/WHUBuildingDataset'

dataset_kind = 'train'

def plot_cov_ellipse(cov, mean, ax, n_std=2.0, **kwargs):
    """
    Plots an n-std covariance ellipse.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)

    from matplotlib.patches import Ellipse
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        **kwargs
    )
    ax.add_patch(ellipse)

def SemanticMaskGenerator(common_path, dataset_kind):

    coco_json = f"{common_path}/annotation/{dataset_kind}.json"
    image_dir = f"{common_path}/{dataset_kind}/images"
    mask_dir = f"{common_path}/{dataset_kind}/gt"

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # ---- Load COCO ----
    with open(coco_json, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]

    # ---- Group annotations by image ----
    anns_per_image = defaultdict(list)
    for ann in annotations:
        anns_per_image[ann["image_id"]].append(ann)

    image_NOann = 0

    # ---- Process ALL images ----
    for image_id, img_info in tqdm(images.items(), desc="Generating masks"):

        height = img_info["height"]
        width = img_info["width"]

        # Initialize black mask (background)
        mask = np.zeros((height, width), dtype=np.uint8)

        # If image has annotations, draw them
        if image_id in anns_per_image:
            for ann in anns_per_image[image_id]:
                segmentation = ann["segmentation"]

                # Polygon
                if isinstance(segmentation, list):
                    for poly in segmentation:
                        poly = np.array(poly).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 255)

                # RLE
                else:
                    rle = coco_mask.frPyObjects(segmentation, height, width)
                    rle_mask = coco_mask.decode(rle)
                    mask[rle_mask > 0] = 255
        else:
            image_NOann += 1  # background-only image

        # ---- Save mask ----
        base_name = os.path.splitext(img_info["file_name"])[0]
        mask_path = os.path.join(mask_dir, base_name + ".tif")
        tiff.imwrite(mask_path, mask)

    print(f"Images without annotations: {image_NOann}")
def NegativeCreation(NEG_RATIO, MAX_TRIES, PORTION):
        
    # -------------------------------------------------
    # PARAMETERS
    # -------------------------------------------------
    NEG_RATIO = 1.0        # RATIO BETWEEN NEGATIVE AND POSITIVE SAMPLES
    MAX_TRIES = 50         # NUMBER OF ATTEMPTS TO CREATE A NEGATIVE SAMPLE
    PORTION = 0.1           # THE MAX PORTION ADMITTED TO BE SAVED AS NEGATIVE SAMPLES
    semantic_mask_dir = f"{common_path}/tiles/{dataset_kind}/gt"
    negative_out_dir = f"{common_path}/instances/negatives/{dataset_kind}"
    
    os.makedirs(negative_out_dir, exist_ok=True)
    os.makedirs(negative_out_dir + '/images', exist_ok=True)
    os.makedirs(negative_out_dir + '/gt', exist_ok=True)
    
    # -------------------------------------------------
    # Prepare size pool from positives
    # -------------------------------------------------
    sizes = list(zip(widths_pos, heights_pos))
    num_negatives = int(len(sizes) * NEG_RATIO)
    
    # All image ids available
    image_ids = list(images.keys())
    
    neg_count = 0
    
    heigths_neg, widths_neg = [], []
    
    # -------------------------------------------------
    # Negative sampling loop
    # -------------------------------------------------
    for i in tqdm(range(num_negatives), desc="Sampling negatives"):
    
        w, h = random.choice(sizes)
    
        for _ in range(MAX_TRIES):
    
            # Pick a random image
            image_id = random.choice(image_ids)
            img_info = images[image_id]
    
            img_path = os.path.join(image_dir, img_info["file_name"])
            mask_path = os.path.join(
                semantic_mask_dir,
                os.path.splitext(img_info["file_name"])[0] + ".tif"
            )
    
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
            mask = tiff.imread(mask_path)
    
            H, W, _ = image.shape
    
            # Skip if crop does not fit
            if w >= W or h >= H:
                continue
    
            heigths_neg.append(h)
            widths_neg.append(w)
            
            MAX_VALUE = (w * h) * PORTION
            # Random top-left corner
            x = random.randint(0, W - w)
            y = random.randint(0, H - h)
    
            mask_crop = mask[y:y+h, x:x+w]
    
            # Accept only pure background
            if mask_crop.sum() <= MAX_VALUE:
    
                rgb_crop = image[y:y+h, x:x+w]
    
                neg_name = f"{neg_count:06d}"
    
                tiff.imwrite(f"{negative_out_dir}/images/{neg_name}.tif", rgb_crop)
                tiff.imwrite(f"{negative_out_dir}/gt/{neg_name}.tif", mask_crop)
    
                neg_count += 1
                break
    
    print(f"Total negatives saved: {neg_count}")
def InstanceMaskGenerator(common_path, dataset_kind, pad_ratio=0.1):

    widths, heights = [], []


    coco_json = f"{common_path}/annotation/{dataset_kind}.json"
    image_dir = f"{common_path}/tiles/{dataset_kind}/images"
    instance_images_dir = f"{common_path}/instances/{dataset_kind}/images"
    instance_gt_dir = f"{common_path}/instances/{dataset_kind}/gt"

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(instance_images_dir, exist_ok=True)
    os.makedirs(instance_gt_dir, exist_ok=True)

    PAD_RATIO = pad_ratio  # context around building (10% of the building size)

    # ---- Load COCO ----
    with open(coco_json, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]

    # ---- Group annotations by image ----
    anns_per_image = defaultdict(list)
    for ann in annotations:
        anns_per_image[ann["image_id"]].append(ann)

    instance_counter = 0

    # ---- Process images with annotations only ----
    for image_id in tqdm(anns_per_image.keys(), desc="Extracting instances"):

        img_info = images[image_id]
        img_path = os.path.join(image_dir, img_info["file_name"])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W, _ = image.shape

        for idx, ann in enumerate(anns_per_image[image_id], start=1):

            # ---- Decode instance mask ----
            mask = np.zeros((H, W), dtype=np.uint8)

            segmentation = ann["segmentation"]

            if isinstance(segmentation, list):  # polygon
                for poly in segmentation:
                    poly = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 255)
            else:  # RLE
                rle = coco_mask.frPyObjects(segmentation, H, W)
                rle_mask = coco_mask.decode(rle)
                # If multiple masks: sum along third axis
                if len(rle_mask.shape) == 3:
                    rle_mask = np.sum(rle_mask, axis=2)
                mask[rle_mask > 0] = 255

            # ---- Bounding box with padding ----
            x, y, w, h = map(int, ann["bbox"])

            heights.append(h)
            widths.append(w)
            
            pad = int(np.clip(max(w, h) * PAD_RATIO, 8, 128))

            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, W)
            y2 = min(y + h + pad, H)

            rgb_crop = image[y1:y2, x1:x2]
            mask_crop = mask[y1:y2, x1:x2]

            # ---- Save ----
            base = os.path.splitext(img_info["file_name"])[0]
            inst_name = f"{base}_{idx:04d}"

            tiff.imwrite(os.path.join(instance_images_dir, inst_name + ".tif"), rgb_crop)
            tiff.imwrite(os.path.join(instance_gt_dir, inst_name + ".tif"), mask_crop)

            instance_counter += 1

    print(f"Total instances saved: {instance_counter}")
    return heights, widths

def Plots(widths, heights, dataset_kind, class_):
    # ---- Statistics ----
    mean_w = np.mean(widths)
    mean_h = np.mean(heights)
    median_w = np.median(widths)
    median_h = np.median(heights)
    
    cov = np.cov(widths, heights)  # 2x2 covariance matrix
    
    print("Mean width :", mean_w)
    print("Mean height:", mean_h)
    print("Median width", median_w)
    print("Median heights", median_h)
    print("Covariance matrix:\n", cov)
    
    widths = np.array(widths)
    heights = np.array(heights)
    
    aspect_ratios = widths / heights
    areas = widths * heights
    
    plt.figure(figsize=(14, 10))
    plt.axis('off')
    
    plt.title(label=f"{dataset_kind}-{class_}", fontsize=30, fontstyle='italic')
    
    # ---------------- Width histogram ----------------
    plt.subplot(2, 2, 1)
    plt.hist(widths, bins=50, color="steelblue", alpha=0.8)
    plt.yscale("log")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Count (log scale)")
    plt.title("Building Width Distribution")
    plt.axvline(mean_w, color="red", linestyle="--", label=f"Mean: {mean_w:.2f}")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # ---------------- Height histogram ----------------
    plt.subplot(2, 2, 2)
    plt.hist(heights, bins=50, color="darkorange", alpha=0.8)
    plt.yscale("log")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Count (log scale)")
    plt.title("Building Height Distribution")
    plt.axvline(mean_h, color="red", linestyle="--", label=f"Mean: {mean_h:.2f}")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # ---------------- Aspect ratio ----------------
    plt.subplot(2, 2, 3)
    plt.hist(aspect_ratios, bins=100, color="seagreen", alpha=0.8)
    plt.yscale("log")
    plt.xlabel("Aspect Ratio (W / H)")
    plt.ylabel("Count (log scale)")
    plt.title("Aspect Ratio Distribution")
    plt.axvline(1.0, color="darkgreen", label="Square")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # ---------------- Width vs Height ----------------
    ax = plt.subplot(2, 2, 4)
    ax.scatter(widths, heights, s=5, alpha=0.3)
    
    # Mean lines
    ax.axvline(mean_w, color="red", linestyle="--", label="Mean width")
    ax.axhline(mean_h, color="blue", linestyle="--", label="Mean height")
    
    # Mean point
    ax.scatter(mean_w, mean_h, color="black", s=80, marker="x", label="Mean")
    
    # Covariance ellipse (2σ)
    plot_cov_ellipse(cov, (mean_w, mean_h), ax, n_std=2,
                     edgecolor="purple", linewidth=2, label="2σ ellipse")
    
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    ax.set_title("Width vs Height (mean & covariance)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(common_path + f'/instances/{class_}-{dataset_kind}_shape_distribution.png')
    plt.close()

#local path on macbook
common_path = '/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/WHUBuildingDataset'

#path of the nas
common_path = '/mnt/nas151/sar/Footprint/datasets/WHUBuildingDataset'

dataset_kind = 'train'


if __name__ == "__main__":

    args = parse_args()

    common_path = args.common_path
    dataset_kind = args.dataset_type
    mode = args.mode
    pad_ratio = args.pad_ratio

    print(f"Mode: {mode}")
    print(f"Dataset: {dataset_kind}")
    print(f"Path: {common_path}")

    if mode == "semantic":
        SemanticMaskGenerator(common_path, dataset_kind)

    elif mode == "instance":
        heights, widths = InstanceMaskGenerator(
            common_path=common_path,
            dataset_kind=dataset_kind,
            pad_ratio=pad_ratio
        )

    elif mode == "stats":
        heights, widths = InstanceMaskGenerator(
            common_path=common_path,
            dataset_kind=dataset_kind,
            pad_ratio=pad_ratio
        )
        Plots(widths, heights, dataset_kind, class_="positive")

    elif mode == "negatives":
        raise NotImplementedError("NegativeCreation requires positive stats first: run 'instance' mode")

    else:
        raise ValueError(f"Unknown mode {mode}")