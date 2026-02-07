import numpy as np
import torch
from safetensors.torch import load_file
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from skimage.exposure import match_histograms
from torchange.models.segment_any_change.segment_anything.utils.amg import MaskData
from torchvision.ops.boxes import batched_nms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegmentAnyChange:
    def __init__(
        self,
        model_cfg="sam2.1_hiera_l.yaml",
        sam2_checkpoint="../models_ckpt/sam2.1_hiera_large.pt",
    ):
        sam2 = build_sam2(
            model_cfg, sam2_checkpoint, device=DEVICE, apply_postprocessing=False
        )
        self.sam2 = sam2.to(DEVICE)

        self.mask_generator = SAM2AutomaticMaskGenerator(sam2)

        self.set_hyperparameters()

        self.embed_data1 = None
        self.embed_data2 = None

    def set_hyperparameters(self, **kwargs):
        self.match_hist = kwargs.get("match_hist", False)
        self.area_thresh = kwargs.get("area_thresh", 0.8)

    def make_mask_generator(self, **kwargs):
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2, **kwargs)

    def extract_image_embedding(self, img1, img2):
        self.mask_generator.predictor.set_image(img1)
        self.embed_data1 = {
            "image_embedding": self.mask_generator.predictor.get_image_embedding(),
            "original_size": img1.shape[:2],
        }
        self.mask_generator.predictor.set_image(img2)
        self.embed_data2 = {
            "image_embedding": self.mask_generator.predictor.get_image_embedding(),
            "original_size": img2.shape[:2],
        }
        return self.embed_data1, self.embed_data2

    def set_cached_embedding(self, embedding):
        data = embedding
        oh, ow = data["original_size"].numpy()
        h, w = data["input_size"]
        self.embed_data1 = {
            "image_embedding": data["t1"].to(DEVICE),
            "original_size": (oh, ow),
        }

        self.embed_data2 = {
            "image_embedding": data["t2"].to(DEVICE),
            "original_size": (oh, ow),
        }
        self.mask_generator.predictor.input_size = (h, w)
        self.mask_generator.predictor.original_size = (oh, ow)

    def load_cached_embedding(self, filepath):
        data = load_file(filepath, device="cpu")
        self.set_cached_embedding(data)

    def clear_cached_embedding(self):
        self.embed_data1 = None
        self.embed_data2 = None
        self.mask_generator.predictor.input_size = None
        self.mask_generator.predictor.original_size = None

    def proposal(self, img1, img2):
        h, w = img1.shape[:2]
        if self.embed_data1 is None:
            self.extract_image_embedding(img1, img2)

        mask_data1 = self.mask_generator._generate_masks(img1)
        mask_data2 = self.mask_generator._generate_masks(img2)
        # mask_data1.filter((mask_data1["areas"] / (h * w)) < self.area_thresh)
        # mask_data2.filter((mask_data2["areas"] / (h * w)) < self.area_thresh)

        return {
            "t1_mask_data": mask_data1,
            "t1_image_embedding": self.embed_data1["image_embedding"],
            "t2_mask_data": mask_data2,
            "t2_image_embedding": self.embed_data2["image_embedding"],
        }

    def bitemporal_match(
        self, t1_mask_data, t1_image_embedding, t2_mask_data, t2_image_embedding
    ) -> MaskData:
        return NotImplementedError

    def forward(self, img1, img2):
        h, w = img1.shape[:2]

        if self.match_hist:
            img2 = match_histograms(image=img2, reference=img1, channel_axis=-1).astype(
                np.uint8
            )

        data = self.proposal(img1, img2)

        changemasks = self.bitemporal_match(**data)

        boxes = changemasks["boxes"]
        scores = changemasks["iou_preds"]

        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes)

        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores)

        boxes = boxes.to(DEVICE)
        scores = scores.to(DEVICE)

        keep = batched_nms(
            boxes.float(),
            scores,
            torch.zeros_like(boxes[:, 0]),
            iou_threshold=0.7,
        )
        changemasks.filter(keep)

        if len(changemasks["rles"]) > 1000:
            scores = changemasks["change_confidence"]
            sorted_scores, _ = torch.sort(scores, descending=True, stable=True)
            keep = scores > sorted_scores[1000]
            changemasks.filter(keep)

        return changemasks, data["t1_mask_data"], data["t2_mask_data"]

    def to_eval_format_predictions(self, cmasks):
        boxes = cmasks["boxes"]
        rle_masks = cmasks["rles"]
        labels = torch.ones(boxes.size(0), dtype=torch.int64)
        scores = cmasks["change_confidence"]
        predictions = {
            "boxes": boxes.to(torch.float32).cpu(),
            "scores": scores.cpu(),
            "labels": labels.cpu(),
            "masks": rle_masks,
        }
        return predictions

    def __call__(self, img1, img2):
        cmasks, t1_masks, t2_masks = self.forward(img1, img2)
        predictions = self.to_eval_format_predictions(cmasks)
        self.clear_cached_embedding()
        return predictions
