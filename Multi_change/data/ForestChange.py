import json
import os
from random import *

import cv2
import numpy as np
import torch

# import cv2 as cv
from imageio import imread
from preprocess_data import encode
from torch.utils.data import DataLoader, Dataset


class ForestChangeDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(
        self,
        data_folder,
        list_path,
        split,
        token_folder=None,
        vocab_file=None,
        max_length=42,
        allow_unk=0,
        transform=None,
        target_size=None,
        img_size=(256, 256),
        max_iters=None,
    ):
        """
        :param data_folder: folder where image files are stored
        :param list_path: folder where the file name-lists of Train/val/test.txt sets are stored
        :param split: one of 'TRAIN', 'VAL', or 'TEST'
        :param token_folder: folder where token files are stored
        :param vocab_file: the name of vocab file
        :param max_length: the maximum length of each caption sentence
        :param allow_unk: whether to allow the tokens have unknow word or not
        :param transform: list of transformations applied to each example for augmentation
        :param target_size: if inflating dataset size, the size of the dataset to sample augmentation from
        :param img_size: the dimensions all images should be returned as
        :param max_iters: the maximum iteration when loading the data
        """
        self.list_path = list_path
        self.split = split
        self.max_length = max_length
        self.transform = transform
        self.img_size = img_size
        self.target_size = target_size

        assert self.split in {"train", "val", "test"}
        self.img_ids = [
            i_id.strip() for i_id in open(os.path.join(list_path + split + ".txt"))
        ]
        if vocab_file is not None:
            with open(os.path.join(list_path + vocab_file + ".json"), "r") as f:
                self.word_vocab = json.load(f)
            self.allow_unk = allow_unk

        self.files = []
        if split == "train":
            for name in self.img_ids:
                img_fileA = os.path.join(
                    data_folder + "/" + split + "/A/" + name.split("-")[0]
                )
                img_fileB = img_fileA.replace("A", "B")

                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_fileA.replace("A", "label"))

                if "-" in name:
                    token_id = name.split("-")[-1]
                else:
                    token_id = None
                if token_folder is not None:
                    token_file = os.path.join(
                        token_folder + name.split(".")[0] + ".txt"
                    )
                else:
                    token_file = None
                self.files.append(
                    {
                        "imgA": imgA,
                        "imgB": imgB,
                        "seg_label": seg_label,
                        "token": token_file,
                        "token_id": token_id,
                        "name": name.split("-")[0],
                    }
                )
        elif split == "val":
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + "/" + split + "/A/" + name)
                img_fileB = img_fileA.replace("A", "B")

                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_fileA.replace("A", "label"))

                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(
                        token_folder + name.split(".")[0] + ".txt"
                    )
                else:
                    token_file = None
                self.files.append(
                    {
                        "imgA": imgA,
                        "imgB": imgB,
                        "seg_label": seg_label,
                        "token": token_file,
                        "token_id": token_id,
                        "name": name,
                    }
                )
        elif split == "test":
            for name in self.img_ids:
                img_fileA = os.path.join(data_folder + "/" + split + "/A/" + name)
                img_fileB = img_fileA.replace("A", "B")

                imgA = imread(img_fileA)
                imgB = imread(img_fileB)
                seg_label = imread(img_fileA.replace("A", "label"))

                token_id = None
                if token_folder is not None:
                    token_file = os.path.join(
                        token_folder + name.split(".")[0] + ".txt"
                    )
                else:
                    token_file = None
                self.files.append(
                    {
                        "imgA": imgA,
                        "imgB": imgB,
                        "seg_label": seg_label,
                        "token": token_file,
                        "token_id": token_id,
                        "name": name,
                    }
                )

        if self.target_size is not None and len(self.files) < self.target_size:
            self._inflate_dataset()

        if max_iters is not None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = (
                self.img_ids * n_repeat
                + self.img_ids[: max_iters - n_repeat * len(self.img_ids)]
            )

    def _inflate_dataset(self):
        original_size = len(self.files)
        needed = self.target_size - original_size

        for i in range(needed):
            original_idx = i % original_size
            original = self.files[original_idx]

            augmented_example = {
                "imgA": original["imgA"].copy(),
                "imgB": original["imgB"].copy(),
                "seg_label": original["seg_label"].copy(),
                "token": original["token"],
                "token_id": original["token_id"],
                "name": f"{original['name']}_aug{i}",
            }

            self.files.append(augmented_example)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]

        imgA = datafiles["imgA"]
        imgB = datafiles["imgB"]
        seg_label = datafiles["seg_label"]
        seg_label[seg_label != 0] = 1

        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)

        if imgA.shape[1] != self.img_size[0] or imgA.shape[2] != self.img_size[1]:
            imgA = cv2.resize(imgA, self.img_size)
            imgB = cv2.resize(imgB, self.img_size)
            seg_label = cv2.resize(seg_label, self.img_size)

        if self.transform:  # transform should contain a normalisation transform
            augmented = self.transform(image=imgA, image_B=imgB)
            imgA = augmented["image"]
            imgB = augmented["image_B"]
        else:
            mean = [0.2267 * 255, 0.29982 * 255, 0.22058 * 255]
            std = [0.0923 * 255, 0.06658 * 255, 0.05681 * 255]
            for i, _ in enumerate(mean):
                imgA[:, :, i] -= mean[i]
                imgA[:, :, i] /= std[i]
                imgB[:, :, i] -= mean[i]
                imgB[:, :, i] /= std[i]

        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)

        if datafiles["token"] is not None:
            with open(datafiles["token"], "r") as caption_file:
                caption_list = json.load(caption_file)

            token_all = np.zeros((len(caption_list), self.max_length), dtype=int)
            token_all_len = np.zeros((len(caption_list), 1), dtype=int)
            for j, tokens in enumerate(caption_list):
                nochange_cap = [
                    "<START>",
                    "the",
                    "scene",
                    "is",
                    "the",
                    "same",
                    "as",
                    "before",
                    "<END>",
                ]
                if self.split == "train" and nochange_cap in caption_list:
                    tokens = nochange_cap
                tokens_encode = encode(
                    tokens, self.word_vocab, allow_unk=self.allow_unk == 1
                )
                token_all[j, : len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)
            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
                j = randint(0, len(caption_list) - 1)
                token = token_all[j]
                token_len = token_all_len[j].item()
        else:
            token_all = np.zeros(1, dtype=int)
            token = np.zeros(1, dtype=int)
            token_len = np.zeros(1, dtype=int)
            token_all_len = np.zeros(1, dtype=int)

        return (
            imgA.copy(),
            imgB.copy(),
            seg_label.copy(),
            token_all.copy(),
            token_all_len.copy(),
            token.copy(),
            np.array(token_len),
            name,
        )
