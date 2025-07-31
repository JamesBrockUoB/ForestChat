import argparse
import json

# 添加特定路径到 Python 解释器的搜索路径中
# sys.path.append('F:\LCY\Change_Agent\Change-Agent-git\Multi_change')
import os.path
import sys

import cv2
import numpy as np
import torch.optim
from genericpath import exists
from griffe import check
from imageio.v2 import imread
from model.model_decoder import DecoderTransformer
from model.model_encoder_att import AttentiveEncoder, Encoder
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


# compute_change_map(path_A, path_B)函数: 生成一个掩膜mask用来表示两个图像之间的变化区域
"""
Args:
    path_A: 图像A的路径
    path_B: 图像B的路径
Returns:
    change_map: 变化区域的掩膜
"""
# def compute_change_mask(path_A, path_B):
#     import cv2
#     import numpy as np
#     img_A = cv2.imread(path_A)
#     img_B = cv2.imread(path_B)
#     change_map = (img_B-img_A).astype(np.uint8)
#     # 阈值化
#     change_map = cv2.cvtColor(change_map, cv2.COLOR_BGR2GRAY)
#     change_map = cv2.threshold(change_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     cv2.imwrite('E:\change_map.png', change_map)
#     return 'I have save the changed mask in E:\change_map.png'

# compute_change_caption(path_A, path_B)函数：生成一个文本用于描述两个图像之间变化
"""
Args:
    path_A: 图像A的路径
    path_B: 图像B的路径
Returns:
    caption: 变化描述文本
"""


class Change_Perception(object):
    def define_args(self, parent_parser=None):

        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        print(script_dir)
        parser = argparse.ArgumentParser(
            description="Remote_Sensing_Image_Change_Interpretation",
            parents=[parent_parser] if parent_parser else [],
            add_help=False,
        )

        parser.add_argument(
            "--data_folder",
            default="./data/Forest-Change-dataset/images",
        )
        parser.add_argument(
            "--list_path",
            default="./data/Forest-Change/",
        )
        parser.add_argument("--vocab_file", default="vocab")
        parser.add_argument(
            "--metadata_file",
            default="metadata",
            help="path of the metadata file for the dataset",
        )
        parser.add_argument("--gpu_id", type=int, default=0)
        parser.add_argument(
            "--checkpoint", default="./models_ckpt/ForestChat_model.pth"
        )
        parser.add_argument("--result_path", default="./predict_results/")
        parser.add_argument("--network", default="segformer-mit_b1")
        parser.add_argument("--encoder_dim", type=int, default=512)
        parser.add_argument("--feat_size", type=int, default=16)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--n_heads", type=int, default=8)
        parser.add_argument("--n_layers", type=int, default=3)
        parser.add_argument("--decoder_n_layers", type=int, default=1)
        parser.add_argument("--feature_dim", type=int, default=512)

        args = parser.parse_args()

        return args

    def __init__(self, parent_parser=None):
        """
        Training and validation.
        """
        args = self.define_args(parent_parser=parent_parser)
        self.args = args
        if "Forest-Change" in args.data_folder:
            self.mean = [0.2267 * 255, 0.29982 * 255, 0.22058 * 255]
            self.std = [0.0923 * 255, 0.06658 * 255, 0.05681 * 255]
        else:
            self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
            self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]

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
        self.encoder_trans = AttentiveEncoder(
            train_stage=None,
            n_layers=args.n_layers,
            feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
            heads=args.n_heads,
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

        imgA = imgA.unsqueeze(0)  # (1, 3, 256, 256)
        imgB = imgB.unsqueeze(0)

        return imgA, imgB

    def generate_change_caption(self, path_A, path_B):
        print("model_infer_change_captioning: start")
        imgA, imgB = self.preprocess(path_A, path_B)
        # Move to GPU, if available
        imgA = imgA.to(DEVICE)
        imgB = imgB.to(DEVICE)
        feat1, feat2 = self.encoder(imgA, imgB)
        feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
        seq = self.decoder.sample(feat1, feat2, k=1)
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

        caption = "there is forest change"
        caption = pred_caption
        print("change captioning:", caption)
        return caption

    def change_detection(self, path_A, path_B, savepath_mask):
        print("model_infer_change_detection: start")
        imgA, imgB = self.preprocess(path_A, path_B)
        # Move to GPU, if available
        imgA = imgA.to(DEVICE)
        imgB = imgB.to(DEVICE)
        feat1, feat2 = self.encoder(imgA, imgB)
        feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
        # for segmentation
        pred_seg = seg_pre.data.cpu().numpy()
        pred_seg = np.argmax(pred_seg, axis=1)
        # 保存图片
        pred = pred_seg[0].astype(np.uint8)

        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        if "Forest-Change" in self.args.data_folder:
            pred_rgb = pred
        else:
            pred_rgb[pred == 1] = [0, 255, 255]
            pred_rgb[pred == 2] = [0, 0, 255]

        cv2.imwrite(savepath_mask, pred_rgb)
        print("model_infer: mask saved in", savepath_mask)

        print("model_infer_change_detection: end")
        return pred  # (256,256,3) or (256, 256) if not rgb
        # return 'change detection successfully. '

    def sac_change_detection(self, path_A, path_B, savepath_mask):
        print("model_infer_change_detection_with_sac: start")
        imgA = imread(path_A)
        imgB = imread(path_B)

        m = AnyChange("vit_h", sam_checkpoint="./models_ckpt/sam_vit_h_4b8939.pth")

        m.make_mask_generator(
            points_per_side=32,
            stability_score_thresh=0.95,
        )

        m.set_hyperparameters(
            change_confidence_threshold=145,
            use_normalized_feature=True,
            bitemporal_match=True,
        )

        mask_data, _, _ = m.forward(imgA, imgB)

        assert isinstance(mask_data, MaskData)
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

        cv2.imwrite(savepath_mask, img_rgb)

        print("model_infer: mask saved in", savepath_mask)

        print("model_infer_change_detection_with_sac: end")
        return img_rgb

    def sac_change_detection_points_of_interest(
        self, path_A, path_B, savepath_mask, xyts
    ):
        print("model_infer_change_detection_with_sac_points_of_interest: start")
        imgA = imread(path_A)
        imgB = imread(path_B)

        m = AnyChange("vit_h", sam_checkpoint="./models_ckpt/sam_vit_h_4b8939.pth")

        m.make_mask_generator(
            points_per_side=32,
            stability_score_thresh=0.85,
        )

        m.set_hyperparameters(
            change_confidence_threshold=165,
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

        cv2.imwrite(savepath_mask, img_rgb)

        print("model_infer: mask saved in", savepath_mask)

        print("model_infer_change_detection_with_sac_points_of_interest: end")
        return img_rgb

    def compute_object_num(self, changed_mask, object):
        print("compute num start")
        # compute the number of connected components
        mask = changed_mask
        mask_cp = 0 * mask.copy()
        if object == "road":
            mask_cp[mask == 1] = 255
        elif object == "building":
            mask_cp[mask == 2] = 255
        lbl = measure.label(mask_cp, connectivity=2)
        props = measure.regionprops(lbl)
        # get bboxes by a for loop
        bboxes = []
        for prop in props:
            # print('Found bbox', prop.bbox, 'area:', prop.area)
            if prop.area > 5:
                bboxes.append([prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]])
        num = len(bboxes)
        # visual
        # mask_array_copy = mask.copy()*255
        # for bbox in bboxes:
        #     print('Found bbox', bbox)
        #     cv2.rectangle(mask_array_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), 2)
        # cv2.namedWindow('findCorners', 0)
        # cv2.resizeWindow('findCorners', 600, 600)
        # cv2.imshow('findCorners', mask_array_copy)
        # cv2.waitKey(0)
        print("Found", num, object)
        print("compute num end")
        # return
        num_str = "Found " + str(num) + " changed " + object
        return num_str

    # design more tool functions:
    def compute_deforestation_percentage(self, mask_path):
        """Calculate percentage from mask with error handling"""
        try:
            img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            total_pixels = img.shape[0] * img.shape[1]
            deforestation_pixels = np.sum(img != 0)
            percentage = round((deforestation_pixels / total_pixels) * 100.0, 2)
            return f"{percentage} percent of the observed area has been affected by deforestation"
        except Exception:
            return "Problem loading the change mask"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remote_Sensing_Image_Change_Interpretation"
    )

    # Custom args for inference
    parser.add_argument("--imgA_path", required=True)
    parser.add_argument("--imgB_path", required=True)
    parser.add_argument("--mask_save_path", required=True)

    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.mask_save_path)):
        os.makedirs(os.path.dirname(args.mask_save_path), exist_ok=True)

    imgA_path = args.imgA_path
    imgB_path = args.imgB_path

    Change_Perception = Change_Perception(parent_parser=parser)
    Change_Perception.generate_change_caption(imgA_path, imgB_path)
    Change_Perception.change_detection(imgA_path, imgB_path, args.mask_save_path)
    Change_Perception.compute_deforestation_percentage(args.mask_save_path)

    base, ext = os.path.splitext(args.mask_save_path)
    sac_mask_filename = f"{base}_sac{ext}"
    Change_Perception.sac_change_detection(imgA_path, imgB_path, sac_mask_filename)
