import argparse
import json
from itertools import islice

import cv2
import torch.optim
from data.ForestChange import ForestChangeDataset
from data.LEVIR_MCI import LEVIRCCDataset
from model.model_decoder import DecoderTransformer
from model.model_encoder_att import AttentiveEncoder, Encoder
from torch.utils import data
from torchange.models.segment_any_change import AnyChange
from torchange.models.segment_any_change.segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    rle_to_mask,
)
from tqdm import tqdm
from utils_tool.metrics import Evaluator
from utils_tool.utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASS = 2  # 3 for LEVIR-MCI


def create_bw_mask(mask_data):
    assert isinstance(mask_data, MaskData)

    # Return blank mask if no segments
    if len(mask_data["rles"]) == 0:
        return np.zeros((256, 256), dtype=np.uint8)  # Default size

    # Get dimensions from first mask
    base_mask = rle_to_mask(mask_data["rles"][0])
    h, w = base_mask.shape

    # Create empty canvas
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # Combine all masks (sorted by area descending)
    sorted_rles = sorted(
        mask_data["rles"], key=lambda x: area_from_rle(x), reverse=True
    )
    for rle in sorted_rles:
        mask = rle_to_mask(rle).astype(np.uint8)
        cv2.bitwise_or(combined_mask, mask, dst=combined_mask)

    # Scale to 0-255 (white=foreground)
    return combined_mask * 255


def save_mask(pred, gt, name, save_path, args):
    # pred value: 0,1,2; map to black, yellow, red
    # gt value: 0,1,2; map to black, yellow, red
    name = name[0]
    evaluator = Evaluator(num_class=NUM_CLASS)
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
    # 保存image_A 和 image_B
    img_A_path = os.path.join(args.data_folder, "test/A", name)
    img_B_path = os.path.join(args.data_folder, "test/B", name)
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
        f"{os.path.basename(snapshot_full_path).replace('.pth', '')}_sac",
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

    # Custom dataloaders
    if args.data_name in ["LEVIR_MCI", "Forest-Change"]:
        dataset = (
            ForestChangeDataset(
                args.data_folder,
                args.list_path,
                "test",
                args.token_folder,
                args.vocab_file,
                0,
                args.allow_unk,
            )
            if "Forest-Change" in args.data_name
            else LEVIRCCDataset(
                args.data_folder,
                args.list_path,
                "test",
                args.token_folder,
                args.vocab_file,
                0,
                args.allow_unk,
            )
        )
        test_loader = data.DataLoader(
            dataset,
            batch_size=args.test_batchsize,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

    # Epochs
    evaluator = Evaluator(num_class=NUM_CLASS)
    with torch.no_grad():
        for ind, (
            imgA,
            imgB,
            seg_label,
            token_all,
            token_all_len,
            _,
            _,
            name,
        ) in enumerate(
            tqdm(test_loader, desc="test_" + " EVALUATING AT BEAM SIZE " + str(1))
        ):
            # Move to GPU, if available
            imgA = imgA.to(DEVICE)
            imgB = imgB.to(DEVICE)

            seg_label = seg_label.cpu().numpy()

            m = AnyChange(
                "vit_h",
                sam_checkpoint="./Multi_change/models_ckpt/sam_vit_h_4b8939.pth",
            )
            m.make_mask_generator(
                points_per_side=16,
                stability_score_thresh=0.95,
            )

            m.set_hyperparameters(
                change_confidence_threshold=155,
                use_normalized_feature=True,
                bitemporal_match=True,
            )

            changemasks, _, _ = m.forward(imgA, imgB)
            pred_seg = create_bw_mask(changemasks)

            # for change detection: save mask?
            if args.save_mask:
                save_mask(pred_seg, seg_label, name, args.result_path, args)
            # Add batch sample into evaluator
            evaluator.add_batch(seg_label, pred_seg)

        # Fast test during the training

        Acc_seg = evaluator.Pixel_Accuracy()
        Acc_class_seg = evaluator.Pixel_Accuracy_Class()
        mIoU_seg, IoU = evaluator.Mean_Intersection_over_Union()
        FWIoU_seg = evaluator.Frequency_Weighted_Intersection_over_Union()
        print(
            "Validation:\n"
            "Acc_seg: {0:.5f}\t"
            "Acc_class_seg: {1:.5f}\t"
            "mIoU_seg: {2:.5f}\t"
            "FWIoU_seg: {3:.5f}\t".format(Acc_seg, Acc_class_seg, mIoU_seg, FWIoU_seg)
        )
        print("IoU:", IoU)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remote_Sensing_Image_Change_Interpretation"
    )

    # Data parameters
    parser.add_argument("--sys", default="linux", help="system win or linux")
    parser.add_argument(
        "--data_folder",
        default="./data/Forest-Change-dataset/images",
        help="folder with image files",
    )
    parser.add_argument(
        "--list_path", default="./data/Forest-Change/", help="path of the data lists"
    )
    parser.add_argument(
        "--data_name", default="Forest-Change", help="base name shared by data files."
    )

    # Test
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id in the training.")
    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="print training/validation stats every __ batches",
    )
    parser.add_argument("--test_batchsize", default=1, help="batch_size for test")
    parser.add_argument("--workers", type=int, default=0, help="for data-loading")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    # save masks and captions?
    parser.add_argument(
        "--save_mask", action="store_false", help="save the result of masks"
    )
    parser.add_argument(
        "--result_path",
        default="./predict_results",
        help="path to save the result of masks and captions",
    )

    args = parser.parse_args()

    main(args)
    main(args)
