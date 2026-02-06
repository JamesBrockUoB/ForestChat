import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from data.ForestChange import ForestChangeDataset
from data.LEVIRMCITrees import LEVIRMCITreesDataset
from mci_model.model_decoder import DecoderTransformer
from mci_model.model_encoder_att import AttentiveEncoder, Encoder
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
        word_vocab = json.load(f)

    with open(os.path.join(args.list_path) + args.metadata_file + ".json", "r") as f:
        max_length = json.load(f)["max_length"]

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

    encoder = Encoder(args.network)

    dims = [32, 64, 160, 256] if "b0" in args.network else [64, 128, 320, 512]
    encoder_trans = AttentiveEncoder(
        train_stage=None,
        n_layers=args.n_layers,
        feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
        heads=args.n_heads,
        num_classes=args.num_classes,
        dims=dims,
        dropout=args.dropout,
    )
    decoder = DecoderTransformer(
        encoder_dim=args.encoder_dim,
        feature_dim=args.feature_dim,
        vocab_size=len(word_vocab),
        max_lengths=max_length,
        word_vocab=word_vocab,
        n_head=args.n_heads,
        n_layers=args.decoder_n_layers,
        dropout=args.dropout,
    )

    encoder.load_state_dict(checkpoint["encoder_dict"])
    encoder_trans.load_state_dict(checkpoint["encoder_trans_dict"], strict=False)
    decoder.load_state_dict(checkpoint["decoder_dict"])
    # Move to GPU, if available
    encoder.eval()
    encoder = encoder.to(DEVICE)
    encoder_trans.eval()
    encoder_trans = encoder_trans.to(DEVICE)
    decoder.eval()
    decoder = decoder.to(DEVICE)

    # Custom dataloaders
    if args.data_name in ["LEVIR-MCI-Trees", "Forest-Change"]:
        dataset = (
            ForestChangeDataset(
                data_folder=args.data_folder,
                list_path=args.list_path,
                split=args.split,
                token_folder=args.token_folder,
                vocab_file=args.vocab_file,
                max_length=max_length,
                allow_unk=args.allow_unk,
                num_classes=args.num_classes,
            )
            if "Forest-Change" in args.data_name
            else LEVIRMCITreesDataset(
                data_folder=args.data_folder,
                list_path=args.list_path,
                split=args.split,
                token_folder=args.token_folder,
                vocab_file=args.vocab_file,
                max_length=max_length,
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
    else:
        raise ValueError("Unknown dataset selected")

    # Epochs
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    evaluator = Evaluator(num_class=args.num_classes)

    total_start = time.time()

    seg_start = time.time()
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
            tqdm(
                test_loader,
                desc="test_SEGMENTATION",
            )
        ):
            imgA = imgA.to(DEVICE)
            imgB = imgB.to(DEVICE)
            if encoder is not None:
                feat1, feat2 = encoder(imgA, imgB)
            feat1, feat2, seg_pre = encoder_trans(feat1, feat2)
            pred_seg = np.argmax(seg_pre.data.cpu().numpy(), axis=1)
            seg_label_np = seg_label.cpu().numpy()
            if args.save_mask:
                save_mask(pred_seg, seg_label_np, name, args.result_path, args)
            evaluator.add_batch(seg_label_np, pred_seg)

    seg_time = time.time() - seg_start

    cap_start = time.time()
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
            tqdm(
                test_loader,
                desc="test_CAPTIONING" + " EVALUATING AT BEAM SIZE " + str(1),
            )
        ):
            imgA = imgA.to(DEVICE)
            imgB = imgB.to(DEVICE)
            token_all = token_all.squeeze(0).to(DEVICE)
            if encoder is not None:
                feat1, feat2 = encoder(imgA, imgB)
            feat1, feat2, _ = encoder_trans(feat1, feat2)
            seq = decoder.sample(feat1, feat2)
            img_token = token_all.tolist()
            img_tokens = list(
                map(
                    lambda c: [
                        w
                        for w in c
                        if w
                        not in {
                            word_vocab["<START>"],
                            word_vocab["<END>"],
                            word_vocab["<NULL>"],
                        }
                    ],
                    img_token,
                )
            )  # remove <start> and pads
            references.append(img_tokens)
            pred_seq = [
                w
                for w in seq
                if w
                not in {
                    word_vocab["<START>"],
                    word_vocab["<END>"],
                    word_vocab["<NULL>"],
                }
            ]
            hypotheses.append(pred_seq)
            assert len(references) == len(hypotheses)
            pred_caption = " ".join(list(word_vocab.keys())[i] for i in pred_seq)

            ref_captions = ""
            for tokens in img_tokens:
                ref_captions += (
                    " ".join(list(word_vocab.keys())[i] for i in tokens) + ".    "
                )

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

    cap_time = time.time() - cap_start

    total_time = time.time() - total_start

    print(f"Total time:        {total_time:.3f} s")
    print(f"Segmentation time: {seg_time:.3f} s")
    print(f"Captioning time:   {cap_time:.3f} s")

    # Fast test during the training

    Acc_seg = evaluator.Pixel_Accuracy()
    Acc_class_seg = evaluator.Pixel_Accuracy_Class()
    mIoU_seg, IoU = evaluator.Mean_Intersection_over_Union()
    FWIoU_seg = evaluator.Frequency_Weighted_Intersection_over_Union()
    Acc_seg = evaluator.Pixel_Accuracy()
    F1_score, F1_class_score = evaluator.F1_Score()
    Precision, Precision_class, Recall, Recall_class = (
        evaluator.Precision_Recall_Class()
    )
    print(
        "Test of Segmentation:\n"
        "Time: {0:.3f}\t"
        "Acc_seg: {1:.5f}\t"
        "Acc_class_seg: {2:.5f}\t"
        "mIoU_seg: {3:.5f}\t"
        "FWIoU_seg: {4:.5f}\t"
        "IoU: {5}\t"
        "F1: {6:.5f}\t"
        "F1_class: {7}\t"
        "Precision: {8:.5f}\t"
        "Precision_class: {9}\t"
        "Recall: {10:.5f}\t"
        "Recall_class: {11}\t".format(
            seg_time,
            Acc_seg,
            Acc_class_seg,
            mIoU_seg,
            FWIoU_seg,
            IoU,
            F1_score,
            F1_class_score,
            Precision,
            Precision_class,
            Recall,
            Recall_class,
        )
    )

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
            cap_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider
        )
    )


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
        "--network",
        default="segformer-mit_b1",
        help="define the backbone encoder to extract features",
    )
    parser.add_argument(
        "--encoder_dim",
        type=int,
        default=512,
        help="the dimension of extracted features using backbone ",
    )
    parser.add_argument(
        "--feat_size",
        type=int,
        default=16,
        help="define the output size of encoder to extract features",
    )
    # Model parameters
    parser.add_argument(
        "--n_heads", type=int, default=8, help="Multi-head attention in Transformer."
    )
    parser.add_argument(
        "--n_layers", type=int, default=3, help="Number of layers in AttentionEncoder."
    )
    parser.add_argument("--decoder_n_layers", type=int, default=1)
    parser.add_argument(
        "--feature_dim", type=int, default=512, help="embedding dimension"
    )
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--split", default="test")

    args = parser.parse_args()

    main(args)
