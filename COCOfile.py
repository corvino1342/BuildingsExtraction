import os
import json
import numpy as np
import cv2
import tifffile as tiff
from collections import defaultdict
from pycocotools import mask as coco_mask
from tqdm import tqdm


def SemanticMaskGenerator(dataset_kind):

    coco_json = f"/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/WHUBuildingDataset/annotation/{dataset_kind}.json"
    image_dir = f"/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/WHUBuildingDataset/{dataset_kind}/images"
    mask_dir = f"/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/WHUBuildingDataset/{dataset_kind}/gt"

    os.makedirs(mask_dir, exist_ok=True)

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

def InstanceMaskGenerator(dataset_kind):

    common_path = f'/Users/corvino/PycharmProjects/BuildingsExtraction/datasets/WHUBuildingDataset'

    coco_json = f"{common_path}/annotation/{dataset_kind}.json"
    image_dir = f"{common_path}/{dataset_kind}/images"
    instance_images_dir = f"{common_path}/instance/{dataset_kind}/images"
    instance_gt_dir = f"{common_path}/instance/{dataset_kind}/gt"

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(instance_images_dir, exist_ok=True)
    os.makedirs(instance_gt_dir, exist_ok=True)

    PAD_RATIO = 0.1  # context around building (10% of the building size)

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

InstanceMaskGenerator(dataset_kind='val')