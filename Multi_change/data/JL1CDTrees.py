"""
DataLoader for JL1-CD-Trees woodland change detection test set.
Simplified version without caption loading.
"""

import os

import cv2
import numpy as np
from imageio import imread
from torch.utils.data import Dataset

NORMALISATION_MEAN = [0.2427 * 255, 0.3016 * 255, 0.1888 * 255]
NORMALISATION_STD = [0.1592 * 255, 0.1403 * 255, 0.1314 * 255]


class JL1CDTreesDataset(Dataset):
    """
    Dataset for JL1-CD-Trees woodland change detection test set.
    """

    def __init__(
        self,
        data_folder,
        split,
        img_size=(256, 256),
        num_classes=2,
        max_percent_samples=None,
    ):
        """
        Args:
            :param data_folder: Path to JL1-CD-Trees root folder (containing A/, B/, label/)
            :param img_size: Target image size (height, width)
            :param num_classes: Number of classes present in the change masks
            :param max_percent_samples: maximum percentage of samples returned by the dataset if running few-shot learning (0-100)
        """
        self.data_folder = data_folder
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.PIXEL_SIZE = 0.5
        self.max_percent_samples = max_percent_samples

        assert self.split in {"train", "val", "test"}

        self.files = []
        img_dir_A = os.path.join(data_folder, split, "A")
        img_dir_B = os.path.join(data_folder, split, "B")
        label_dir = os.path.join(data_folder, split, "label")

        if not os.path.exists(img_dir_A):
            img_dir_A = os.path.join(data_folder, "A")
            img_dir_B = os.path.join(data_folder, "B")
            label_dir = os.path.join(data_folder, "label")

        img_names = sorted([f for f in os.listdir(img_dir_A) if f.endswith(".png")])

        for name in img_names:
            img_file_A = os.path.join(img_dir_A, name)
            img_file_B = os.path.join(img_dir_B, name)
            label_file = os.path.join(label_dir, name)

            if (
                os.path.exists(img_file_A)
                and os.path.exists(img_file_B)
                and os.path.exists(label_file)
            ):
                self.files.append(
                    {
                        "imgA": img_file_A,
                        "imgB": img_file_B,
                        "label": label_file,
                        "name": name,
                    }
                )
        if max_percent_samples is not None:
            max_samples = round(len(self.files) * self.max_percent_samples / 100)
            print(f"Limiting {split} split to {max_samples} samples")
            self.files = self.files[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]

        imgA = imread(datafiles["imgA"])
        imgB = imread(datafiles["imgB"])
        seg_label = imread(datafiles["label"])

        seg_label[seg_label != 0] = 1

        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)
        seg_label = np.asarray(seg_label, np.float32)

        if imgA.shape[0] != self.img_size[0] or imgA.shape[1] != self.img_size[1]:
            imgA = cv2.resize(imgA, (self.img_size[1], self.img_size[0]))
            imgB = cv2.resize(imgB, (self.img_size[1], self.img_size[0]))
            seg_label = cv2.resize(seg_label, (self.img_size[1], self.img_size[0]))

        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)

        for i, _ in enumerate(NORMALISATION_MEAN):
            imgA[:, :, i] -= NORMALISATION_MEAN[i]
            imgA[:, :, i] /= NORMALISATION_STD[i]
            imgB[:, :, i] -= NORMALISATION_MEAN[i]
            imgB[:, :, i] /= NORMALISATION_STD[i]

        return {
            "imgA": imgA.copy(),
            "imgB": imgB.copy(),
            "label": seg_label.copy(),
            "name": name,
        }
