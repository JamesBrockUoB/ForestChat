import argparse

import torch.optim
import wandb
from torch.utils import data
from torchange.models.segment_any_change import AnyChange
from tqdm import tqdm
from utils_tool.metrics import Evaluator
from utils_tool.utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASS = 2  # 3 for LEVIR-MCI

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/mIoU", "goal": "maximize"},
    "parameters": {
        "change_confidence_threshold": {"min": 0.5, "max": 0.95},
        "points_per_side": {"values": [8, 16, 24, 32]},
        "stability_score_thresh": {"min": 0.85, "max": 0.98},
        "bitemporal_match": {"values": [True, False]},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 3},
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

        dataset = load_images_sac(args.data_folder, "test")
        self.val_loader = data.DataLoader(
            dataset,
            batch_size=args.val_batchsize,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        self.evaluator = Evaluator(num_class=NUM_CLASS)

    def configure_model(self, config):
        self.model = AnyChange("vit_h", sam_checkpoint="./models/sam_vit_h_4b8939.pth")
        self.model.make_mask_generator(
            points_per_side=config.points_per_side,
            stability_score_thresh=config.stability_score_thresh,
        )
        self.model.set_hyperparameters(
            change_confidence_threshold=config.change_confidence_threshold,
            use_normalized_feature=config.use_normalized_feature,
            bitemporal_match=config.bitemporal_match,
        )

    # One epoch's validation
    def validation(self, config):
        val_start_time = time.time()
        self.configure_model(config)

        # Batches
        for imgA, imgB, seg_label, _ in enumerate(
            tqdm(self.val_loader, desc="val_" + "EVALUATING AT BEAM SIZE " + str(1))
        ):
            # Move to GPU, if available
            imgA = imgA.to(DEVICE).numpy()
            imgB = imgB.to(DEVICE).numpy()

            seg_label = seg_label.cpu().numpy()

            imgA = imgA.squeeze(0)
            imgB = imgB.squeeze(0)

            if imgA.shape[0] == 3:
                imgA = imgA.transpose(1, 2, 0)
                imgB = imgB.transpose(1, 2, 0)

            changemasks, _, _ = m.forward(imgA, imgB)
            pred_seg = create_bw_mask(changemasks)

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
    parser.add_argument("--sys", default="win", help="system win or linux")
    parser.add_argument(
        "--data_folder",
        default="./data/Forest-Change-dataset/images",
        help="folder with data files",
    )
    parser.add_argument(
        "--list_path", default="./data/Forest-Change/", help="path of the data lists"
    )
    parser.add_argument(
        "--data_name", default="Forest-Change", help="base name shared by data files."
    )

    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id in the training.")

    parser.add_argument(
        "--print_freq",
        type=int,
        default=5,
        help="print training/validation stats every __ batches",
    )
    parser.add_argument(
        "--increased_val_data_size",
        type=int,
        default=None,
        help="if you provide a number, it will increase the validation dataset size to match the number",
    )
    parser.add_argument("--workers", type=int, default=4, help="for data-loading")
    # Validation
    parser.add_argument(
        "--val_batchsize", type=int, default=1, help="batch_size for validation"
    )
    args = parser.parse_args()

    sweep_id = wandb.sweep(
        SWEEP_CONFIG, project="forest-chat-sac", entity="your-wandb-username"
    )

    # Create searcher instance
    searcher = SACHyperparameterSearcher(args)

    def sweep_run():
        with wandb.init() as run:
            # Run validation with current config
            metrics = searcher.run_validation(wandb.config)

            # Log metrics to W&B
            wandb.log(metrics)

            # Optionally save best config
            if searcher.best_config == wandb.config:
                wandb.run.summary["best"] = True
                wandb.run.summary["best_mIoU"] = metrics["val/mIoU"]

    # Run the sweep
    wandb.agent(sweep_id, function=sweep_run, count=25)

    # Print final best configuration
    print("\n=== Best Configuration ===")
    print(f"mIoU: {searcher.best_mIoU:.4f}")
    print(searcher.best_config)
