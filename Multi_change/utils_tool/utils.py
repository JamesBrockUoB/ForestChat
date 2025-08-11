import os
import time

import albumentations as A
import cv2
import numpy as np
import torch
from eval_func.bleu.bleu import Bleu
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor
from eval_func.rouge.rouge import Rouge
from skimage.io import imread
from skimage.transform import resize
from torchange.models.segment_any_change.segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    rle_to_mask,
)


def create_binary_mask_sac(mask_data):
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


def load_images_sac(data_folder, split, target_size=(256, 256)):
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
        label[label != 0] = 1

        images.append((imgA.copy(), imgB.copy(), label.copy(), img_name))

    return images


def get_image_transforms():
    transform = A.Compose(
        [
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=1.0
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=1.0
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 3), sigma_limit=0.5, p=1.0),
                    A.GaussNoise(
                        std_range=(0.02, 0.06),
                        mean_range=(0, 0),
                        per_channel=True,
                        noise_scale_factor=1.0,
                        p=1.0,
                    ),
                ],
                p=0.3,
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


def compute_class_weights(files, num_classes):
    """
    Compute class weights based on pixel frequency in segmentation masks to improve learning.

    Returns:
        torch.Tensor of shape (N,) where N is the number of classes
    """
    pixel_counts = np.zeros(num_classes, dtype=np.int64)

    for data in files:
        seg_label = data["seg_label"]

        if seg_label.ndim == 3:
            seg_mask = seg_label[:, :, 0]
        else:
            seg_mask = seg_label

        for c in range(num_classes):
            pixel_counts[c] += np.sum(seg_mask == c)

    total_pixels = pixel_counts.sum()
    pixel_freqs = pixel_counts / total_pixels

    class_weights = 1.0 / (pixel_freqs + 1e-6)
    class_weights = class_weights * (num_classes / class_weights.sum())

    return torch.tensor(class_weights, dtype=torch.float32)


def save_checkpoint(
    args,
    data_name,
    epoch,
    encoder,
    encoder_feat,
    decoder,
    encoder_optimizer,
    encoder_feat_optimizer,
    decoder_optimizer,
    best_bleu4,
):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {
        "epoch": epoch,
        "best_bleu-4": best_bleu4,
        "encoder": encoder,
        "encoder_feat": encoder_feat,
        "decoder": decoder,
        "encoder_optimizer": encoder_optimizer,
        "encoder_feat_optimizer": encoder_feat_optimizer,
        "decoder_optimizer": decoder_optimizer,
    }
    # filename = 'checkpoint_' + data_name + '_' + args.network + '.pth.tar'
    path = args.savepath  #'./models_checkpoint/mymodel/3-times/'
    if os.path.exists(path) == False:
        os.makedirs(path)
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    torch.save(state, os.path.join(path, "BEST_" + data_name))

    # torch.save(state, os.path.join(path, 'checkpoint_' + data_name +'_epoch_'+str(epoch) + '.pth.tar'))


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
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
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


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]["lr"],))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_file_str():
    ISOTIMEFORMAT = "%Y-%m-%d-%H-%M-%S"
    string = "{}".format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string  # + '-{}'.format(random.randint(1, 10000))


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write("{:}\n".format(print_string))
    log.flush()
    log.flush()
    log.flush()
