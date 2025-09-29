import argparse
import json
import os

import cv2
import numpy as np
import torch.optim
from torch.utils import data
from torchange.models.segment_any_change import AnyChange
from tqdm import tqdm
from utils_tool.metrics import Evaluator
from utils_tool.utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_mask(pred, gt, name, save_path, split, args):
    # pred value: 0,1; map to black, red
    # gt value: 0,1; map to black, red

    name = name[0]
    evaluator = Evaluator(num_class=args.num_classes)
    evaluator.add_batch(gt, pred)
    mIoU_seg, IoU = evaluator.Mean_Intersection_over_Union()
    Miou_str = round(mIoU_seg, 4)
    # Miou_str save in json file named score
    json_name = os.path.join(save_path, "score.json")
    if not os.path.exists(json_name):
        with open(json_name, "a+") as f:
            key = name.split(".")[0]
            json.dump({f"{key}": {"MIoU": Miou_str}}, f)
        f.close()
    else:
        with open(os.path.join(save_path, "score.json"), "r") as file:
            data = json.load(file)
            key = name.split(".")[0]
            data[key] = {"MIoU": Miou_str}
        # write to json file
        with open(os.path.join(save_path, "score.json"), "w") as file:
            json.dump(data, file)
        file.close()

    # save mask
    pred = pred[0].astype(np.uint8)
    gt = gt[0].astype(np.uint8)
    pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    pred_rgb[pred == 1] = [0, 255, 255]
    pred_rgb[pred == 2] = [0, 0, 255]
    gt_rgb[gt == 1] = [0, 255, 255]
    gt_rgb[gt == 2] = [0, 0, 255]

    cv2.imwrite(os.path.join(save_path, name.split(".")[0] + f"_mask.png"), pred_rgb)
    cv2.imwrite(os.path.join(save_path, name.split(".")[0] + "_gt.png"), gt_rgb)

    pred_bin = (pred > 0).astype(bool)
    gt_bin = (gt > 0).astype(bool)
    diff_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    diff_rgb[pred_bin & gt_bin] = [255, 255, 0]
    diff_rgb[pred_bin & ~gt_bin] = [255, 0, 0]
    diff_rgb[~pred_bin & gt_bin] = [0, 255, 0]
    diff_rgb = cv2.cvtColor(diff_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(
        os.path.join(save_path, name.split(".")[0] + "_pred_diff.png"), diff_rgb
    )

    img_A_path = os.path.join(args.data_folder, split, "A", name)
    img_B_path = os.path.join(args.data_folder, split, "B", name)
    img_A = cv2.imread(img_A_path)
    img_B = cv2.imread(img_B_path)
    cv2.imwrite(os.path.join(save_path, name.split(".")[0] + "_A.png"), img_A)
    cv2.imwrite(os.path.join(save_path, name.split(".")[0] + "_B.png"), img_B)


def main(args):
    """
    Testing.
    """

    args.result_path = os.path.join(
        args.result_path,
        "anychange_model",
    )
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path, exist_ok=True)
    else:
        print("result_path exists!")
        # clear folder
        for root, dirs, files in os.walk(args.result_path):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    dataset = load_images_anychange(args.data_folder, args.split)
    test_loader = data.DataLoader(
        dataset,
        batch_size=args.test_batchsize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Epochs
    evaluator = Evaluator(num_class=args.num_classes)

    with torch.no_grad():
        for batch in tqdm(
            test_loader, desc="test_" + " EVALUATING AT BEAM SIZE " + str(1)
        ):
            imgA, imgB, seg_label, name = batch
            # Move to GPU, if available
            imgA = imgA.to(DEVICE).numpy()
            imgB = imgB.to(DEVICE).numpy()

            seg_label = seg_label.cpu().numpy()

            imgA = imgA.squeeze(0)
            imgB = imgB.squeeze(0)

            if imgA.shape[0] == 3:
                imgA = imgA.transpose(1, 2, 0)
                imgB = imgB.transpose(1, 2, 0)

            m = AnyChange(
                "vit_h",
                sam_checkpoint=args.anychange_network_path,
            )
            m.make_mask_generator(
                points_per_side=16,
                stability_score_thresh=0.94,
            )

            m.set_hyperparameters(
                change_confidence_threshold=150,
                use_normalized_feature=True,
                bitemporal_match=True,
            )

            changemasks, _, _ = m.forward(imgA, imgB)
            pred_seg = create_binary_mask_anychange(changemasks)

            # for change detection: save mask?
            if args.save_mask:
                save_mask(pred_seg, seg_label, name, args.result_path, args.split, args)
            # Add batch sample into evaluator
            evaluator.add_batch(seg_label, pred_seg)

        # Fast test during the training

        Acc_seg = evaluator.Pixel_Accuracy()
        Acc_class_seg = evaluator.Pixel_Accuracy_Class()
        mIoU_seg, IoU = evaluator.Mean_Intersection_over_Union()
        FWIoU_seg = evaluator.Frequency_Weighted_Intersection_over_Union()
        Acc_seg = evaluator.Pixel_Accuracy()
        F1_score, F1_class_score = evaluator.F1_Score()
        print(
            "Validation:\n"
            "Acc_seg: {0:.5f}\t"
            "Acc_class_seg: {1:.5f}\t"
            "mIoU_seg: {2:.5f}\t"
            "FWIoU_seg: {3:.5f}\t"
            "IoU: {4}\t"
            "F1: {5:.5f}\t"
            "F1_class: {6}\t".format(
                Acc_seg,
                Acc_class_seg,
                mIoU_seg,
                FWIoU_seg,
                IoU,
                F1_score,
                F1_class_score,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remote_Sensing_Image_Change_Interpretation"
    )

    # Data parameters
    parser.add_argument(
        "--data_folder",
        default="./data/Forest-Change-dataset/images",
        help="folder with image files",
    )
    parser.add_argument(
        "--anychange_network_path",
        default="./models_ckpt/sam_vit_h_4b8939.pth",
        help="path of the backbone architecture used by AnyChange",
    )

    # Test
    parser.add_argument("--test_batchsize", default=1, help="batch_size for test")
    parser.add_argument("--workers", type=int, default=0, help="for data-loading")
    parser.add_argument(
        "--save_mask", type=str2bool, default=True, help="save the result of masks"
    )
    parser.add_argument(
        "--result_path",
        default="./predict_results",
        help="path to save the result of masks and captions",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_classes", default=2)

    args = parser.parse_args()

    main(args)
