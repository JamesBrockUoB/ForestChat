import argparse
import os
import time

import numpy as np
import torch.optim
import wandb
from dotenv import load_dotenv
from torch.utils import data
from torchange.models.segment_any_change import AnyChange
from tqdm import tqdm
from utils_tool.metrics import Evaluator
from utils_tool.utils import *

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASS = 2  # 3 for LEVIR-MCI

SWEEP_CONFIG = {
    "name": "SAC Hyperparameter Search",
    "method": "bayes",
    "metric": {"name": "val/mIoU", "goal": "maximize"},
    "parameters": {
        "change_confidence_threshold": {
            "values": [int(x) for x in np.arange(130, 180, 5)]
        },
        "points_per_side": {"values": [8, 16, 24, 32]},
        "stability_score_thresh": {
            "values": [round(x, 2) for x in np.arange(0.85, 0.98, 0.01)]
        },
    },
    "early_terminate": {"type": "hyperband", "min_iter": 3, "eta": 2},
}


class SACHyperparameterSearcher(object):
    def __init__(self, args):
        """
        Validation on different hyperparameter search settings.
        """
        self.args = args
        self.best_mIoU = 0
        self.best_config = None
        self.model = None

        dataset = load_images_sac(args.data_folder, "val")
        self.val_loader = data.DataLoader(
            dataset,
            batch_size=args.val_batchsize,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        self.evaluator = Evaluator(num_class=NUM_CLASS)

    # One epoch's validation
    def validation(self, config):
        val_start_time = time.time()

        # Batches
        for batch in tqdm(
            self.val_loader, desc="val_" + "EVALUATING AT BEAM SIZE " + str(1)
        ):
            imgA, imgB, seg_label, _ = batch
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
                sam_checkpoint=self.args.sac_network_path,
            )
            m.make_mask_generator(
                points_per_side=config.points_per_side,
                stability_score_thresh=config.stability_score_thresh,
            )

            m.set_hyperparameters(
                change_confidence_threshold=config.change_confidence_threshold,
                use_normalized_feature=True,
                bitemporal_match=True,
            )

            changemasks, _, _ = m.forward(imgA, imgB)
            pred_seg = create_binary_mask_sac(changemasks)

            self.evaluator.add_batch(seg_label, pred_seg)

        val_time = time.time() - val_start_time

        metrics = {
            "Validation Time": val_time,
            "val/mIoU": self.evaluator.Mean_Intersection_over_Union()[0],
            "val/Acc": self.evaluator.Pixel_Accuracy(),
            "val/Acc_class": self.evaluator.Pixel_Accuracy_Class(),
            "val/FWIoU": self.evaluator.Frequency_Weighted_Intersection_over_Union(),
            **config,  # Log all hyperparameters
        }

        # Track best configuration
        if metrics["val/mIoU"] > self.best_mIoU:
            self.best_mIoU = metrics["val/mIoU"]
            self.best_config = config
            print(f"New best config: mIoU={self.best_mIoU:.4f}")
            print(config)

        return metrics


if __name__ == "__main__":
    wandb.login()

    parser = argparse.ArgumentParser(
        description="Remote_Sensing_Image_Change_Interpretation"
    )

    # Data parameters
    parser.add_argument(
        "--data_folder",
        default="./data/Forest-Change-dataset/images",
        help="folder with data files",
    )
    # Validation
    parser.add_argument(
        "--val_batchsize", type=int, default=1, help="batch_size for validation"
    )
    parser.add_argument(
        "--sac_network_path",
        default="./models_ckpt/sam_vit_h_4b8939.pth",
        help="path of the backbone architecture used by SAC",
    )
    args = parser.parse_args()

    sweep_id = wandb.sweep(
        SWEEP_CONFIG,
        project="forest-chat-sac",
        entity=os.environ.get("WANDB_USERNAME"),
    )

    # Create searcher instance
    searcher = SACHyperparameterSearcher(args)

    def sweep_run():
        with wandb.init() as run:
            # Run validation with current config
            run.config.update(
                {
                    "sweep_method": SWEEP_CONFIG["method"],
                    "sweep_metric": SWEEP_CONFIG["metric"],
                    "sweep_early_terminate": SWEEP_CONFIG.get("early_terminate", {}),
                }
            )
            metrics = searcher.validation(wandb.config)

            # Log metrics to W&B
            wandb.log(metrics)

            # Optionally save best config
            if searcher.best_config == wandb.config:
                wandb.run.summary["best"] = True
                wandb.run.summary["best_mIoU"] = metrics["val/mIoU"]

    # Run the sweep
    wandb.agent(sweep_id, function=sweep_run, count=20)

    # Print final best configuration
    print("\n=== Best Configuration ===")
    print(f"mIoU: {searcher.best_mIoU:.4f}")
    print(searcher.best_config)
    print(searcher.best_config)
