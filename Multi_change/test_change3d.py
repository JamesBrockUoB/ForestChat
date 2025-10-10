import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from change3d.trainer import Trainer
from data.ForestChange import ForestChangeDataset
from data.LEVIR_MCI import LEVIRCCDataset
from einops import rearrange
from torch.utils import data
from tqdm import tqdm
from utils_tool.metrics import Evaluator
from utils_tool.utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_mask(pred, gt, name, save_path, args):
    # pred value: 0,1,2; map to black, yellow, red
    # gt value: 0,1,2; map to black, yellow, red
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

    img_A_path = os.path.join(args.data_folder, args.split, "A", name)
    img_B_path = os.path.join(args.data_folder, args.split, "B", name)
    img_A = cv2.imread(img_A_path)
    img_B = cv2.imread(img_B_path)
    cv2.imwrite(os.path.join(save_path, name.split(".")[0] + "_A.png"), img_A)
    cv2.imwrite(os.path.join(save_path, name.split(".")[0] + "_B.png"), img_B)


def save_captions(pred_caption, ref_caption, hypotheses, references, name, save_path):
    name = name[0]
    # return 0
    score_dict = get_eval_score([references], [hypotheses])
    Bleu_4 = score_dict["Bleu_4"]
    Bleu_4_str = round(Bleu_4, 4)
    Bleu_3 = score_dict["Bleu_3"]
    Bleu_3_str = round(Bleu_3, 4)

    # read JSON
    with open(os.path.join(save_path, "score.json"), "r") as file:
        data = json.load(file)
        key = name.split(".")[0]
        data[key]["Bleu_3"] = Bleu_3_str
        data[key]["Bleu_4"] = Bleu_4_str
    with open(os.path.join(save_path, "score.json"), "w") as file:
        json.dump(data, file)
    file.close()

    with open(os.path.join(save_path, name.split(".")[0] + f"_cap.txt"), "w") as f:
        f.write("pred_caption: " + pred_caption + "\n")
        f.write("ref_caption: " + ref_caption + "\n")


def main(args):
    """
    Testing.
    """

    with open(os.path.join(args.list_path + args.vocab_file + ".json"), "r") as f:
        args.word_vocab = json.load(f)

    args.vocab_size = len(args.word_vocab)

    with open(os.path.join(args.list_path) + args.metadata_file + ".json", "r") as f:
        args.max_length = json.load(f)["max_length"]

    # Load checkpoint
    snapshot_full_path = args.checkpoint
    checkpoint = torch.load(snapshot_full_path, map_location=DEVICE)

    args.result_path = os.path.join(
        args.result_path, os.path.basename(snapshot_full_path).replace(".pth", "")
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

    model = Trainer(args).to(DEVICE)

    model.encoder.load_state_dict(checkpoint["encoder_dict"])
    model.decoder_cd.load_state_dict(checkpoint["decoder_cd_dict"])
    model.decoder_cc.load_state_dict(checkpoint["decoder_cc_dict"])
    # Move to GPU, if available
    model.encoder.eval()
    model.encoder = model.encoder.to(DEVICE)
    model.decoder_cd.eval()
    model.decoder_cd = model.decoder_cd.to(DEVICE)
    model.decoder_cc.eval()
    model.decoder_cc = model.decoder_cc.to(DEVICE)

    # Custom dataloaders
    if args.data_name in ["LEVIR_MCI", "Forest-Change"]:
        nochange_list = [
            "the scene is the same as before ",
            "there is no difference ",
            "the two scenes seem identical ",
            "no change has occurred ",
            "almost nothing has changed ",
        ]
        dataset = (
            ForestChangeDataset(
                data_folder=args.data_folder,
                list_path=args.list_path,
                split=args.split,
                token_folder=args.token_folder,
                vocab_file=args.vocab_file,
                max_length=args.max_length,
                allow_unk=args.allow_unk,
                num_classes=args.num_classes,
            )
            if "Forest-Change" in args.data_name
            else LEVIRCCDataset(
                data_folder=args.data_folder,
                list_path=args.list_path,
                split=args.split,
                token_folder=args.token_folder,
                vocab_file=args.vocab_file,
                max_length=args.max_length,
                allow_unk=args.allow_unk,
                num_classes=args.num_classes,
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
    test_start_time = time.time()
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    change_references = list()
    change_hypotheses = list()
    nochange_references = list()
    nochange_hypotheses = list()
    change_acc = 0
    nochange_acc = 0
    evaluator = Evaluator(num_class=args.num_classes)
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
            token_all = token_all.squeeze(0).to(DEVICE)
            # Forward prop
            seg_pre, encoder_out = model(imgA, imgB)
            encoder_out = rearrange(encoder_out, "b c h w -> (h w) b c")

            # for segmentation
            pred_seg = seg_pre.data.cpu().numpy()
            seg_label = seg_label.cpu().numpy()
            pred_seg = np.argmax(pred_seg, axis=1)

            # for change detection: save mask?
            if args.save_mask:
                save_mask(pred_seg, seg_label, name, args.result_path, args)

            # Add batch sample into evaluator
            evaluator.add_batch(seg_label, pred_seg)

            # for captioning
            seq = model.decoder_cc.sample_beam(encoder_out, k=args.beam_size)

            # --- reference and hypothesis lists ---
            img_caps = token_all.tolist()
            img_captions = [
                [
                    w
                    for w in c
                    if w
                    not in {
                        args.word_vocab["<START>"],
                        args.word_vocab["<END>"],
                        args.word_vocab["<NULL>"],
                    }
                ]
                for c in img_caps
            ]
            references.append(img_captions)

            hyp = [
                w
                for w in seq
                if w
                not in {
                    args.word_vocab["<START>"],
                    args.word_vocab["<END>"],
                    args.word_vocab["<NULL>"],
                }
            ]
            hypotheses.append(hyp)
            assert len(references) == len(hypotheses)

            pred_caption = ""
            ref_caption = ""
            for i in hyp:
                pred_caption += (list(args.word_vocab.keys())[i]) + " "
            ref_caption = ""
            for i in img_captions[0]:
                ref_caption += (list(args.word_vocab.keys())[i]) + " "
            ref_captions = ""
            for i in img_captions:
                for j in i:
                    ref_captions += (list(args.word_vocab.keys())[j]) + " "
                ref_captions += ".    "
            # for captioning: save captions?
            if args.save_caption:
                save_captions(
                    pred_caption,
                    ref_captions,
                    hypotheses[-1],
                    references[-1],
                    name,
                    args.result_path,
                )
            if ref_caption in nochange_list:
                nochange_references.append(img_captions)
                nochange_hypotheses.append(hyp)
                if pred_caption in nochange_list:
                    nochange_acc = nochange_acc + 1
            else:
                change_references.append(img_captions)
                change_hypotheses.append(hyp)
                if pred_caption not in nochange_list:
                    change_acc = change_acc + 1

        test_time = time.time() - test_start_time

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

        # Calculate evaluation scores
        print("len(nochange_references):", len(nochange_references))
        print("len(change_references):", len(change_references))

        if len(nochange_references) > 0:
            print("nochange_metric:")
            nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
            Bleu_1 = nochange_metric["Bleu_1"]
            Bleu_2 = nochange_metric["Bleu_2"]
            Bleu_3 = nochange_metric["Bleu_3"]
            Bleu_4 = nochange_metric["Bleu_4"]
            Meteor = nochange_metric["METEOR"]
            Rouge = nochange_metric["ROUGE_L"]
            Cider = nochange_metric["CIDEr"]
            print(
                "BLEU-1: {0:.5f}\t"
                "BLEU-2: {1:.5f}\t"
                "BLEU-3: {2:.5f}\t"
                "BLEU-4: {3:.5f}\t"
                "Meteor: {4:.5f}\t"
                "Rouge: {5:.5f}\t"
                "Cider: {6:.5f}\t".format(
                    Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider
                )
            )
            print("nochange_acc:", nochange_acc / len(nochange_references))
        if len(change_references) > 0:
            print("change_metric:")
            change_metric = get_eval_score(change_references, change_hypotheses)
            Bleu_1 = change_metric["Bleu_1"]
            Bleu_2 = change_metric["Bleu_2"]
            Bleu_3 = change_metric["Bleu_3"]
            Bleu_4 = change_metric["Bleu_4"]
            Meteor = change_metric["METEOR"]
            Rouge = change_metric["ROUGE_L"]
            Cider = change_metric["CIDEr"]
            print(
                "BLEU-1: {0:.5f}\t"
                "BLEU-2: {1:.5f}\t"
                "BLEU-3: {2:.5f}\t"
                "BLEU-4: {3:.5f}\t"
                "Meteor: {4:.5f}\t"
                "Rouge: {5:.5f}\t"
                "Cider: {6:.5f}\t".format(
                    Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider
                )
            )
            print("change_acc:", change_acc / len(change_references))

        score_dict = get_eval_score(references, hypotheses)
        Bleu_1 = score_dict["Bleu_1"]
        Bleu_2 = score_dict["Bleu_2"]
        Bleu_3 = score_dict["Bleu_3"]
        Bleu_4 = score_dict["Bleu_4"]
        Meteor = score_dict["METEOR"]
        Rouge = score_dict["ROUGE_L"]
        Cider = score_dict["CIDEr"]
        print(
            "Test of Captioning:\n"
            "Time: {0:.3f}\t"
            "BLEU-1: {1:.5f}\t"
            "BLEU-2: {2:.5f}\t"
            "BLEU-3: {3:.5f}\t"
            "BLEU-4: {4:.5f}\t"
            "Meteor: {5:.5f}\t"
            "Rouge: {6:.5f}\t"
            "Cider: {7:.5f}\t".format(
                test_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider
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
        "--list_path", default="./data/Forest-Change/", help="path of the data lists"
    )
    parser.add_argument(
        "--token_folder",
        default="./data/Forest-Change/tokens/",
        help="folder with token files",
    )
    parser.add_argument("--vocab_file", default="vocab", help="path of the data lists")
    parser.add_argument(
        "--metadata_file",
        default="metadata",
        help="path of the metadata file for the dataset",
    )
    parser.add_argument(
        "--allow_unk",
        type=str2bool,
        default=True,
        help="if unknown token is allowed",
    )
    parser.add_argument(
        "--data_name", default="Forest-Change", help="base name shared by data files."
    )

    # Test
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id in the training.")
    parser.add_argument(
        "--checkpoint",
        default="./models_ckpt/Forest-Change_model.pth",
        help="path to checkpoint",
    )
    parser.add_argument("--test_batchsize", default=1, help="batch_size for test")
    parser.add_argument("--workers", type=int, default=0, help="for data-loading")
    # save masks and captions?
    parser.add_argument(
        "--save_mask", type=str2bool, default=True, help="save the result of masks"
    )
    parser.add_argument(
        "--save_caption",
        type=str2bool,
        default=True,
        help="save the result of captions",
    )
    parser.add_argument(
        "--result_path",
        default="./predict_results",
        help="path to save the result of masks and captions",
    )
    # backbone parameters
    parser.add_argument(
        "--n_head", type=int, default=8, help="Multi-head attention in Transformer."
    )
    parser.add_argument(
        "--pretrained",
        default="models_ckpt/X3D_L.pyth",
        type=str,
        help="Path to pretrained weight",
    )
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--decoder_n_layers", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--num_perception_frame",
        type=int,
        default=1,
        help="Number of perception frames",
    )
    parser.add_argument(
        "--beam_size", type=int, default=1, help="Beam size for beam search."
    )
    parser.add_argument(
        "--in_height", type=int, default=256, help="Height of RGB image"
    )
    parser.add_argument("--in_width", type=int, default=256, help="Width of RGB image")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--split", default="test")

    args = parser.parse_args()

    main(args)
