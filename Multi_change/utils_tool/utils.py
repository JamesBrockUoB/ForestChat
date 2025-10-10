import argparse
import os
import time

import albumentations as A
import cv2
import numpy as np
from data.ForestChange import ForestChangeDataset
from data.LEVIR_MCI import LEVIRCCDataset
from eval_func.bleu.bleu import Bleu
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
from eval_func.rouge.rouge import Rouge
from skimage.io import imread
from torch.utils.data import DataLoader
from torchange.models.segment_any_change.segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    rle_to_mask,
)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_binary_mask_anychange(mask_data):
    assert isinstance(mask_data, MaskData)

    # Return blank mask if no segments (now with batch dim)
    if len(mask_data["rles"]) == 0:
        return np.zeros((1, 256, 256), dtype=np.uint8)  # Shape (1, 256, 256)

    # Get dimensions from first mask
    base_mask = rle_to_mask(mask_data["rles"][0])
    h, w = base_mask.shape

    # Create empty canvas
    combined_mask = np.zeros((1, h, w), dtype=np.uint8)

    # Combine all masks (sorted by area descending)
    sorted_rles = sorted(
        mask_data["rles"], key=lambda x: area_from_rle(x), reverse=True
    )
    for rle in sorted_rles:
        mask = rle_to_mask(rle).astype(np.uint8)
        combined_mask[0, mask > 0] = 1

    return combined_mask


def load_images_anychange(data_folder, split, target_size=(256, 256)):
    folder = os.path.join(data_folder, split)
    images = []

    a_dir = os.path.join(folder, "A")
    for img_name in sorted(os.listdir(a_dir)):
        if not img_name.endswith((".png", ".jpg", ".jpeg")):
            continue

        # Load image pair and label
        imgA = cv2.resize(imread(os.path.join(a_dir, img_name)), target_size)
        imgB = cv2.resize(imread(os.path.join(folder, "B", img_name)), target_size)
        label = cv2.resize(imread(os.path.join(folder, "label", img_name)), target_size)

        if len(label.shape) == 3 and label.shape[2] == 3:
            # Convert to grayscale
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # Binarize the mask (values > 0 become 1)
        label = (label > 0).astype(np.uint8)

        images.append((imgA.copy(), imgB.copy(), label.copy(), img_name))

    return images


def get_image_transforms():
    transform = A.Compose(
        [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0
                    ),
                    A.RandomGamma(gamma_limit=(90, 115), p=1.0),
                    A.HueSaturationValue(
                        hue_shift_limit=(-10, 10),
                        sat_shift_limit=(-15, 15),
                        val_shift_limit=0,
                    ),
                ],
                p=0.7,
            ),
            A.OneOf(
                [
                    A.AdvancedBlur(blur_limit=(3, 5), noise_limit=(0.9, 1.1), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.2), p=1.0),
                ],
                p=0.4,
            ),
            A.Normalize(
                mean=[0.2267 * 255, 0.29982 * 255, 0.22058 * 255],
                std=[0.0923 * 255, 0.06658 * 255, 0.05681 * 255],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        additional_targets={"image_B": "image"},
    )
    return transform


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.reshape(-1, 1).expand_as(ind))
    correct_total = correct.reshape(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    hypo = [
        [" ".join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]
    ]
    ref = [
        [" ".join(reft) for reft in reftmp]
        for reftmp in [
            [[str(x) for x in reft] for reft in reftmp] for reftmp in references
        ]
    ]
    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        (
            method.extend(method_i)
            if isinstance(method_i, list)
            else method.append(method_i)
        )
        # print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict


def time_file_str():
    ISOTIMEFORMAT = "%Y-%m-%d-%H-%M-%S"
    string = "{}".format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string  # + '-{}'.format(random.randint(1, 10000))


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write("{:}\n".format(print_string))
    log.flush()


def build_dataloaders(args, max_length):
    datasets = []
    for split in ["train", "val"]:
        if "Forest-Change" in args.data_name:
            dataset = ForestChangeDataset(
                data_folder=args.data_folder,
                list_path=args.list_path,
                split=split,
                token_folder=args.token_folder,
                vocab_file=args.vocab_file,
                max_length=max_length,
                allow_unk=args.allow_unk,
                transform=(
                    get_image_transforms()
                    if (args.augment and split == "train")
                    else None
                ),
                max_iters=(
                    args.increased_train_data_size
                    if split == "train"
                    else args.increased_val_data_size
                ),
                num_classes=args.num_classes,
            )
        else:  # LEVIR_MCI
            dataset = LEVIRCCDataset(
                data_folder=args.data_folder,
                list_path=args.list_path,
                split=split,
                token_folder=args.token_folder,
                vocab_file=args.vocab_file,
                max_length=max_length,
                allow_unk=args.allow_unk,
                num_classes=args.num_classes,
            )
        datasets.append(dataset)

    train_dataset_size = len(datasets[0])
    train_loader = DataLoader(
        datasets[0],
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets[1],
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_dataset_size, train_loader, val_loader
