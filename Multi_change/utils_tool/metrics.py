import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        # print("Precision(CPA):", Acc)
        Acc = np.nanmean(Acc)  # MPA
        return Acc

    def Recall_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        # print("Recall:", Acc)
        Recall = np.nanmean(Acc)
        return Recall

    def Mean_Intersection_over_Union(self):
        cm = self.confusion_matrix
        intersection = np.diag(cm)
        union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection

        IoU = intersection / np.maximum(union, 1e-6)  # avoid division by zero
        MIoU = np.nanmean(IoU)

        # Format IoU values for each class into a readable string
        IoU_per_class_str = "  ".join([f"{iou:.4f}" for iou in IoU])

        return MIoU, IoU_per_class_str

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
