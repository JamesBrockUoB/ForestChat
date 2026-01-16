import argparse
import json
import os

import cv2
import numpy as np
import torch
from genericpath import exists
from griffe import check
from imageio.v2 import imread
from mci_model.model_decoder import DecoderTransformer
from mci_model.model_encoder_att import AttentiveEncoder, Encoder
from scipy.ndimage import distance_transform_edt
from skimage import measure
from skimage.segmentation import find_boundaries
from torchange.models.segment_any_change import AnyChange
from torchange.models.segment_any_change.segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    box_xyxy_to_xywh,
    rle_to_mask,
)
from utils_tool.utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset configurations
DATASET_CONFIGS = {
    "Forest-Change": {
        "data_folder": "./data/Forest-Change-dataset/images",
        "list_path": "./data/Forest-Change/",
        "checkpoint": "./models_ckpt/Forest-Change_model.pth",
        "num_classes": 2,
        "pixel_area": 30,
        "mean": [0.2267 * 255, 0.29982 * 255, 0.22058 * 255],
        "std": [0.0923 * 255, 0.06658 * 255, 0.05681 * 255],
    },
    "LEVIR-MCI-Trees": {
        "data_folder": "./data/LEVIR-MCI-Trees-dataset/images",
        "list_path": "./data/LEVIR-MCI-Trees/",
        "checkpoint": "./models_ckpt/LEVIR-MCI-Trees_model.pth",
        "num_classes": 3,
        "pixel_area": 0.5,
        "mean": [0.39073 * 255, 0.38623 * 255, 0.32989 * 255],
        "std": [0.15329 * 255, 0.14628 * 255, 0.13648 * 255],
    },
}


class Change_Perception(object):
    def define_args(self, parent_parser=None, dataset_name="Forest-Change"):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        print(script_dir)

        parser = argparse.ArgumentParser(
            description="Remote_Sensing_Image_Change_Interpretation",
            parents=[parent_parser] if parent_parser else [],
            add_help=False,
        )

        # Get dataset-specific config
        config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["Forest-Change"])

        parser.add_argument(
            "--dataset_name",
            default=dataset_name,
            choices=list(DATASET_CONFIGS.keys()),
            help="Name of the dataset to use",
        )
        parser.add_argument(
            "--data_folder",
            default=config["data_folder"],
        )
        parser.add_argument(
            "--list_path",
            default=config["list_path"],
        )
        parser.add_argument("--vocab_file", default="vocab")
        parser.add_argument(
            "--metadata_file",
            default="metadata",
            help="path of the metadata file for the dataset",
        )
        parser.add_argument("--gpu_id", type=int, default=0)
        parser.add_argument("--checkpoint", default=config["checkpoint"])
        parser.add_argument("--result_path", default="./predict_results/")
        parser.add_argument("--network", default="segformer-mit_b1")
        parser.add_argument("--encoder_dim", type=int, default=512)
        parser.add_argument("--feat_size", type=int, default=16)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--n_heads", type=int, default=8)
        parser.add_argument("--n_layers", type=int, default=3)
        parser.add_argument("--decoder_n_layers", type=int, default=1)
        parser.add_argument("--feature_dim", type=int, default=512)
        parser.add_argument("--num_classes", type=int, default=config["num_classes"])

        args = parser.parse_args()
        return args

    def __init__(self, parent_parser=None, dataset_name="Forest-Change"):
        """
        Training and validation.

        Args:
            parent_parser: Parent argument parser (optional)
            dataset_name: Name of the dataset to use ("Forest-Change" or "LEVIR-MCI-Trees").
                         Defaults to "Forest-Change" if not specified.
        """
        args = self.define_args(parent_parser=parent_parser, dataset_name=dataset_name)
        self.args = args
        self.dataset_name = dataset_name

        # Load dataset-specific configuration
        config = DATASET_CONFIGS[dataset_name]
        self.pixel_area = config["pixel_area"]
        self.mean = config["mean"]
        self.std = config["std"]

        with open(os.path.join(args.list_path + args.vocab_file + ".json"), "r") as f:
            self.word_vocab = json.load(f)

        with open(
            os.path.join(args.list_path) + args.metadata_file + ".json", "r"
        ) as f:
            self.max_length = json.load(f)["max_length"]

        # Load checkpoint
        snapshot_full_path = args.checkpoint

        checkpoint = torch.load(snapshot_full_path, map_location=DEVICE)

        self.encoder = Encoder(args.network)

        dims = [32, 64, 160, 256] if "mit_b0" in args.network else [64, 128, 320, 512]
        self.encoder_trans = AttentiveEncoder(
            train_stage=None,
            n_layers=args.n_layers,
            feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
            heads=args.n_heads,
            num_classes=args.num_classes,
            dims=dims,
            dropout=args.dropout,
        )
        self.decoder = DecoderTransformer(
            encoder_dim=args.encoder_dim,
            feature_dim=args.feature_dim,
            vocab_size=len(self.word_vocab),
            max_lengths=self.max_length,
            word_vocab=self.word_vocab,
            n_head=args.n_heads,
            n_layers=args.decoder_n_layers,
            dropout=args.dropout,
        )

        self.encoder.load_state_dict(checkpoint["encoder_dict"])
        self.encoder_trans.load_state_dict(
            checkpoint["encoder_trans_dict"], strict=False
        )
        self.decoder.load_state_dict(checkpoint["decoder_dict"])

        # Move to GPU, if available
        self.encoder.eval()
        self.encoder = self.encoder.to(DEVICE)
        self.encoder_trans.eval()
        self.encoder_trans = self.encoder_trans.to(DEVICE)
        self.decoder.eval()
        self.decoder = self.decoder.to(DEVICE)

    def preprocess(self, path_A, path_B):
        imgA = imread(path_A)
        imgB = imread(path_B)
        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)

        if imgA.shape[1] != 256 or imgA.shape[2] != 256:
            imgA = cv2.resize(imgA, (256, 256))
            imgB = cv2.resize(imgB, (256, 256))

        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)

        for i, _ in enumerate(self.mean):
            imgA[i, :, :] -= self.mean[i]
            imgA[i, :, :] /= self.std[i]
            imgB[i, :, :] -= self.mean[i]
            imgB[i, :, :] /= self.std[i]

        imgA = torch.FloatTensor(imgA)
        imgB = torch.FloatTensor(imgB)

        imgA = imgA.unsqueeze(0)
        imgB = imgB.unsqueeze(0)

        return imgA, imgB

    def generate_change_caption(self, path_A, path_B):
        print("model_infer_change_captioning: start")
        imgA, imgB = self.preprocess(path_A, path_B)
        imgA = imgA.to(DEVICE)
        imgB = imgB.to(DEVICE)
        feat1, feat2 = self.encoder(imgA, imgB)
        feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
        seq = self.decoder.sample(feat1, feat2)
        pred_seq = [
            w
            for w in seq
            if w
            not in {
                self.word_vocab["<START>"],
                self.word_vocab["<END>"],
                self.word_vocab["<NULL>"],
            }
        ]
        pred_caption = ""
        for i in pred_seq:
            pred_caption += (list(self.word_vocab.keys())[i]) + " "

        caption = pred_caption
        print("change captioning:", caption)
        return caption

    def change_detection(self, path_A, path_B, savepath_mask):
        print("model_infer_change_detection: start")
        imgA, imgB = self.preprocess(path_A, path_B)
        imgA = imgA.to(DEVICE)
        imgB = imgB.to(DEVICE)
        feat1, feat2 = self.encoder(imgA, imgB)
        feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
        pred_seg = seg_pre.data.cpu().numpy()
        pred_seg = np.argmax(pred_seg, axis=1)
        pred = pred_seg[0].astype(np.uint8)

        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_rgb[pred == 1] = [0, 255, 255]
        pred_rgb[pred == 2] = [0, 0, 255]

        cv2.imwrite(savepath_mask, pred_rgb)
        print("model_infer: mask saved in", savepath_mask)
        print("model_infer_change_detection: end")
        return pred

    def anychange_change_detection(
        self, path_A, path_B, savepath_mask, process_mask=True
    ):
        print("model_infer_change_detection_with_anychange: start")
        imgA = imread(path_A)
        imgB = imread(path_B)

        m = AnyChange("vit_h", sam_checkpoint="./models_ckpt/sam_vit_h_4b8939.pth")
        m.make_mask_generator(points_per_side=16, stability_score_thresh=0.95)
        m.set_hyperparameters(
            change_confidence_threshold=155,
            use_normalized_feature=True,
            bitemporal_match=True,
        )

        mask_data, _, _ = m.forward(imgA, imgB)
        assert isinstance(mask_data, MaskData)

        if process_mask:
            img = create_binary_mask_anychange(mask_data)[0]
            img_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img_rgb[img == 1] = [0, 255, 255]
        else:
            anns = []
            for idx in range(len(mask_data["rles"])):
                ann_i = {
                    "segmentation": rle_to_mask(mask_data["rles"][idx]),
                    "area": area_from_rle(mask_data["rles"][idx]),
                }
                if "boxes" in mask_data._stats:
                    ann_i["bbox"] = box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist()
                anns.append(ann_i)

            if len(anns) == 0:
                print("No masks to save.")
                return

            sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
            H, W = sorted_anns[0]["segmentation"].shape
            img = np.ones((H, W, 4), dtype=np.float32)
            img[:, :, 3] = 0

            for ann in sorted_anns:
                m = ann["segmentation"]
                boundary = find_boundaries(m)
                color_mask = np.concatenate([np.random.random(3), [0.35]])
                color_boundary = np.array([0.0, 1.0, 1.0, 0.8])
                img[m] = color_mask
                img[boundary] = color_boundary

            img_rgb = (img[:, :, :3] * 255).astype(np.uint8)

        img_rgb = cv2.resize(img_rgb, (256, 256))
        cv2.imwrite(savepath_mask, img_rgb)
        print("model_infer: mask saved in", savepath_mask)
        print("model_infer_change_detection_with_anychange: end")
        return img_rgb

    def anychange_change_detection_points_of_interest(
        self, path_A, path_B, savepath_mask, xyts, process_mask=True
    ):
        print("model_infer_change_detection_with_anychange_points_of_interest: start")
        imgA = imread(path_A)
        imgB = imread(path_B)

        m = AnyChange("vit_h", sam_checkpoint="./models_ckpt/sam_vit_h_4b8939.pth")

        m.make_mask_generator(
            points_per_side=16,
            stability_score_thresh=0.85,
        )

        m.set_hyperparameters(
            change_confidence_threshold=155,
            use_normalized_feature=True,
            bitemporal_match=True,
            object_sim_thresh=70,
        )

        if len(xyts) == 1:
            temporal = xyts[0][-1]
            xy = xyts[0][:2]
            mask_data = m.single_point_match(
                img1=imgA, img2=imgB, temporal=temporal, xy=xy
            )
        else:
            mask_data = m.multi_points_match(img1=imgA, img2=imgB, xyts=xyts)

        assert isinstance(mask_data, MaskData)

        if process_mask:
            img = create_binary_mask_anychange(mask_data)[0]
            img_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img_rgb[img == 1] = [0, 255, 255]
        else:
            anns = []

            for idx in range(len(mask_data["rles"])):
                ann_i = {
                    "segmentation": rle_to_mask(mask_data["rles"][idx]),
                    "area": area_from_rle(mask_data["rles"][idx]),
                }
                if "boxes" in mask_data._stats:
                    ann_i["bbox"] = box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist()
                anns.append(ann_i)

            if len(anns) == 0:
                print("No masks to save.")
                return

            sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
            H, W = sorted_anns[0]["segmentation"].shape
            img = np.ones((H, W, 4), dtype=np.float32)
            img[:, :, 3] = 0  # alpha channel

            for ann in sorted_anns:
                m = ann["segmentation"]
                boundary = find_boundaries(m)
                color_mask = np.concatenate([np.random.random(3), [0.35]])  # RGBA
                color_boundary = np.array([0.0, 1.0, 1.0, 0.8])  # cyan boundaries

                img[m] = color_mask
                img[boundary] = color_boundary

            img_rgb = (img[:, :, :3] * 255).astype(np.uint8)

        img_rgb = cv2.resize(img_rgb, (256, 256))
        cv2.imwrite(savepath_mask, img_rgb)

        print("model_infer: mask saved in", savepath_mask)

        print("model_infer_change_detection_with_anychange_points_of_interest: end")
        return img_rgb

    def compute_object_num(self, changed_mask, object):
        print("model_infer_compute_object_num: start")

        # Normalize mask to binary (handle different input formats)
        mask = changed_mask.copy()
        if len(mask.shape) == 3:
            # For RGB masks, check if ANY channel is non-zero
            mask = np.any(mask > 0, axis=2).astype(np.uint8) * 255

        # Create binary mask based on object type
        mask_cp = np.zeros_like(mask, dtype=np.uint8)

        if object in ["road", "deforestation patches", "all changes"]:
            # Any non-zero pixel is considered changed
            mask_cp[mask > 0] = 255
        elif object == "building":
            # Specifically look for value 2
            mask_cp[mask == 2] = 255

        lbl = measure.label(mask_cp, connectivity=2)
        props = measure.regionprops(lbl)

        # Count patches with area > 5 pixels
        bboxes = []
        for prop in props:
            if prop.area > 5:
                bboxes.append([prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]])

        num = len(bboxes)
        num_str = f"Found {num} changed {object}"
        print(num_str)
        print("model_infer_compute_object_num: end")
        return num_str

    def compute_change_percentage(self, changed_mask):
        print("model_infer_compute_percentage_change: start")

        mask = changed_mask.copy()
        if len(mask.shape) == 3:
            # For RGB masks, check if ANY channel is non-zero (preserves full change area)
            mask = np.any(mask > 0, axis=2).astype(np.uint8) * 255

        total_pixels = mask.shape[0] * mask.shape[1]
        changed_pixels = np.sum(mask > 0)  # Any non-zero value is changed
        percentage = round((changed_pixels / total_pixels) * 100.0, 2)
        percentage_str = f"{percentage} percent of the observed area has been affected"
        print(percentage_str)
        print("model_infer_compute_percentage_change: end")
        return percentage_str

    def compute_patch_metrics(self, changed_mask, object, pixel_area=1.0):
        print("model_infer_compute_change_patch_metrics: start")

        mask = changed_mask.copy()
        if len(mask.shape) == 3:
            # For RGB masks, check if ANY channel is non-zero
            mask = np.any(mask > 0, axis=2).astype(np.uint8) * 255

        mask_cp = np.zeros_like(mask, dtype=np.uint8)

        if object in ["road", "deforestation patches", "all changes"]:
            mask_cp[mask > 0] = 255
        elif object == "building":
            mask_cp[mask == 2] = 255

        lbl = measure.label(mask_cp, connectivity=2)
        props = measure.regionprops(lbl)

        areas = [p.area * pixel_area for p in props if p.area > 5]
        if not areas:
            print("model_infer_compute_change_patch_metrics: end")
            return {"Number of patches": 0}

        perimeters = [p.perimeter for p in props if p.area > 5]
        compactness = [
            (4 * np.pi * a) / (p**2 + 1e-6) for a, p in zip(areas, perimeters)
        ]

        metrics = {
            "Number of patches": len(areas),
            "Total change area (m^2)": round(sum(areas), 2),
            "Mean patch area (m^2)": round(np.mean(areas), 2),
            "Median patch area (m^2)": round(np.median(areas), 2),
            "Largest patch area (m^2)": round(max(areas), 2),
            "Patch area coefficient of variation": round(
                np.std(areas) / (np.mean(areas) + 1e-6), 4
            ),
            "Mean compactness": round(np.mean(compactness), 4),
            "Compactness coefficient of variation": round(
                np.std(compactness) / (np.mean(compactness) + 1e-6), 4
            ),
        }

        print("model_infer_compute_change_patch_metrics: end")
        return metrics

    def compute_linearity_metrics(self, changed_mask, object):
        print("model_infer_compute_change_patch_linearity: start")

        mask = changed_mask.copy()
        if len(mask.shape) == 3:
            # For RGB masks, check if ANY channel is non-zero
            mask = np.any(mask > 0, axis=2).astype(np.uint8) * 255

        mask_cp = np.zeros_like(mask, dtype=np.uint8)

        if object in ["road", "deforestation patches", "all changes"]:
            mask_cp[mask > 0] = 255
        elif object == "building":
            mask_cp[mask == 2] = 255

        lbl = measure.label(mask_cp, connectivity=2)
        props = measure.regionprops(lbl)

        elongations = []
        orientations = []

        for p in props:
            if p.area > 5 and p.minor_axis_length > 0:
                elongations.append(p.major_axis_length / p.minor_axis_length)
                orientations.append(p.orientation)

        if not elongations:
            print("model_infer_compute_change_patch_linearity: end")
            return {
                "Mean elongation": 0.0,
                "High elongation ratio": 0.0,
                "Orientation std": 0.0,
            }

        metrics = {
            "Mean elongation": round(np.mean(elongations), 4),
            "High elongation ratio": round(
                float(np.mean(np.array(elongations) > 3)), 4
            ),
            "Orientation std": round(np.std(orientations), 4),
        }

        print("model_infer_compute_change_patch_linearity: end")
        return metrics

    def compute_edge_core_change(self, changed_mask, object, base_fraction=0.2):
        print("model_infer_compute_change_patch_edge_core: start")

        mask = changed_mask.copy()
        if len(mask.shape) == 3:
            # For RGB masks, check if ANY channel is non-zero
            mask = np.any(mask > 0, axis=2).astype(np.uint8) * 255

        mask_cp = np.zeros_like(mask, dtype=np.uint8)

        if object in ["road", "deforestation patches", "all changes"]:
            mask_cp[mask > 0] = 255
        elif object == "building":
            mask_cp[mask == 2] = 255

        lbl = measure.label(mask_cp, connectivity=2)
        props = measure.regionprops(lbl)

        total_edge = 0
        total_core = 0
        total_pixels = 0

        for p in props:
            if p.area <= 5:  # Skip small patches
                continue

            patch_mask = (lbl == p.label).astype(np.uint8)
            patch_area = patch_mask.sum()

            if patch_area == 0:
                continue

            total_pixels += patch_area

            # Adaptive edge thickness
            adaptive_thresh = max(
                1,
                int(
                    base_fraction
                    * min(
                        p.bbox[2] - p.bbox[0],
                        p.bbox[3] - p.bbox[1],
                    )
                ),
            )

            # Distance transform
            distance = distance_transform_edt(patch_mask)

            # Edge vs core (only count pixels within the patch)
            edge_pixels = np.sum((distance > 0) & (distance <= adaptive_thresh))
            core_pixels = np.sum(distance > adaptive_thresh)

            total_edge += edge_pixels
            total_core += core_pixels

        print("model_infer_compute_change_patch_edge_core: end")

        if total_pixels == 0:
            return {
                "Edge loss ratio": 0.0,
                "Core loss ratio": 0.0,
            }

        return {
            "Edge loss ratio": round(total_edge / total_pixels, 4),
            "Core loss ratio": round(total_core / total_pixels, 4),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remote_Sensing_Image_Change_Interpretation"
    )

    parser.add_argument("--imgA_path", required=True)
    parser.add_argument("--imgB_path", required=True)
    parser.add_argument("--mask_save_path", required=True)
    parser.add_argument(
        "--dataset_name",
        default="Forest-Change",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to use",
    )

    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.mask_save_path)):
        os.makedirs(os.path.dirname(args.mask_save_path), exist_ok=True)

    imgA_path = args.imgA_path
    imgB_path = args.imgB_path

    Change_Perception = Change_Perception(
        parent_parser=parser, dataset_name=args.dataset_name
    )
    Change_Perception.generate_change_caption(imgA_path, imgB_path)
    mask = Change_Perception.change_detection(imgA_path, imgB_path, args.mask_save_path)
    Change_Perception.compute_change_percentage(mask)
    Change_Perception.compute_object_num(mask, "deforestation patches")
    Change_Perception.compute_patch_metrics(
        mask, "deforestation patches", Change_Perception.pixel_area
    )

    base, ext = os.path.splitext(args.mask_save_path)
    anychange_mask_filename = f"{base}_anychange{ext}"
    Change_Perception.anychange_change_detection(
        imgA_path, imgB_path, anychange_mask_filename
    )
