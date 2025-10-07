import argparse
import gc
import json
import os
import random
import time

import numpy as np
import torch
import wandb
from change3d.trainer import Trainer as Change3DModel
from change3d.utils import BCEDiceLoss, adjust_learning_rate, load_checkpoint
from utils_tool.metrics import Evaluator
from utils_tool.utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Change3DTrainer(object):
    def __init__(self, args):
        self.args = args
        self.run = wandb.init(
            project="forest-chat",
            config={
                "dataset_name": args.data_name,
                "train_batchsize": args.train_batchsize,
                "val_batchsize": args.val_batchsize,
                "num_epochs": args.num_epochs,
                "patience": args.patience,
                "decoder_lr": args.decoder_lr,
                "num_classes": args.num_classes,
                "augment": args.augment,
                "num_perception_frame": args.num_perception_frame,
            },
        )

        random_str = str(random.randint(10, 100))
        name = "change3d_" + time_file_str() + f"_det_{random_str}"
        self.args.savepath = os.path.join(args.savepath, name)
        os.makedirs(self.args.savepath, exist_ok=True)
        self.log = open(os.path.join(self.args.savepath, f"{name}.log"), "w")
        print_log(f"=>dataset: {args.data_name}", self.log)
        print_log(f"=>decoder_lr: {args.decoder_lr}", self.log)
        print_log(f"=>num_epochs: {args.num_epochs}", self.log)
        print_log(f"=>train_batchsize: {args.train_batchsize}", self.log)

        self.best_bleu4 = 0.3  # BLEU-4 score right now
        self.MIou = 0.3
        self.Sum_Metric = 0.3
        self.start_epoch = 0

        # Prepare vocabulary/meta if present (for compatibility with train_mci)
        with open(os.path.join(args.list_path + args.vocab_file + ".json"), "r") as f:
            self.word_vocab = json.load(f)

        with open(
            os.path.join(args.list_path + args.metadata_file + ".json"), "r"
        ) as f:
            meta = json.load(f)
            self.max_length = meta.get("max_length", 41)

        # Datasets and loaders (aligned with train_mci)
        self.train_dataset_size, self.train_loader, self.val_loader = build_dataloaders(args, self.max_length)

        # Build model
        model = Change3DModel(self.args)

        self.evaluator = Evaluator(num_class=args.num_classes)

        # Optimizer and LR scheduler
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.decoder_lr,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=1.0
        )

        # Tracking best
        self.best_metric = 0.0  # track best F1
        self.best_epoch = 0
        self.best_model_path = None


    def training(self, epoch):
        self.model.train()
        hist = np.zeros((self.args.num_epochs * len(self.train_loader), 3))
        index_i = 0

        for batch_idx, batch in enumerate(self.train_loader):
            start_time = time.time()
            imgA, imgB, seg_label = (
                batch[0].to(DEVICE).float(),
                batch[1].to(DEVICE).float(),
                batch[2].to(DEVICE),
            )

            # Forward
            pred = self.model.update_bcd(imgA, imgB)
            # Ensure shapes match BCE loss expectation
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            target = seg_label.float()
            if target.dim() == 3:
                target = target.unsqueeze(1)
            loss = self.criterion_det(pred, target)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics for training (binary accuracy from thresholded preds)
            with torch.no_grad():
                bin_pred = (pred > 0.5).long().squeeze(1)
                correct = (bin_pred == seg_label.long()).float().mean().item()

            hist[index_i, 0] = time.time() - start_time
            hist[index_i, 1] = loss.item()
            hist[index_i, 2] = correct
            index_i += 1

            log_now = (
                index_i % self.args.print_freq == 0 and self.args.print_freq > 1
            ) or self.args.print_freq == 1
            if log_now:
                if self.args.print_freq > 1:
                    s, e = index_i - self.args.print_freq, index_i - 1
                    bt, tl, acc = (
                        hist[s:e, 0].mean() * self.args.print_freq,
                        hist[s:e, 1].mean(),
                        hist[s:e, 2].mean(),
                    )
                else:
                    bt, tl, acc = (
                        hist[index_i - 1, 0],
                        hist[index_i - 1, 1],
                        hist[index_i - 1, 2],
                    )

                print_log(
                    "Training Epoch: [{:}][{:}/{:}]\tBatch Time: {:.3f}\tLoss: {:.4f}\tAcc: {:.3f}".format(
                        epoch, batch_idx, len(self.train_loader), bt, tl, acc
                    ),
                    self.log,
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "batch_id": batch_idx,
                        "det_train_loss": tl,
                        "det_train_acc": acc,
                    }
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Step LR per epoch (same as train_mci)
        self.lr_scheduler.step()

    @torch.no_grad()
    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        val_start_time = time.time()
        losses = []

        for batch in self.val_loader:
            imgA, imgB, seg_label = (
                batch[0].to(DEVICE).float(),
                batch[1].to(DEVICE).float(),
                batch[2].to(DEVICE),
            )
            pred = self.model.update_bcd(imgA, imgB)
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            target = seg_label.float()
            if target.dim() == 3:
                target = target.unsqueeze(1)
            loss = self.criterion_det(pred, target)
            losses.append(loss.item())

            bin_pred = (pred > 0.5).long().squeeze(1).cpu().numpy()
            gt = seg_label.cpu().numpy()
            self.evaluator.add_batch(gt, bin_pred)

        val_time = time.time() - val_start_time
        Acc_seg = self.evaluator.Pixel_Accuracy()
        Acc_class_seg = self.evaluator.Pixel_Accuracy_Class()
        mIoU_seg, IoU_per_class = self.evaluator.Mean_Intersection_over_Union()
        FWIoU_seg = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        F1_score, F1_per_class = self.evaluator.F1_Score()
        loss_val = float(np.mean(losses)) if len(losses) > 0 else 0.0

        print_log(
            "Validation:\nTime: {:.3f}\tLoss: {:.5f}\tAcc: {:.5f}\tAcc_class: {:.5f}\tmIoU: {:.5f}\tFWIoU: {:.5f}\tF1: {:.5f}".format(
                val_time,
                loss_val,
                Acc_seg,
                Acc_class_seg,
                mIoU_seg,
                FWIoU_seg,
                F1_score,
            ),
            self.log,
        )
        wandb.log(
            {
                "epoch": epoch,
                "val_loss": loss_val,
                "acc_seg_val": Acc_seg,
                "acc_class_seg_val": Acc_class_seg,
                "mIoU_val": mIoU_seg,
                "FWIoU_val": FWIoU_seg,
                "F1_val": F1_score,
            }
        )

        # Save best model based on F1
        if F1_score >= self.best_metric:
            self.best_metric = F1_score
            self.best_epoch = epoch
            state = {
                "state_dict": self.model.state_dict(),
                "epoch": epoch,
                "F1": F1_score,
            }
            model_name = f"{self.args.data_name}_change3d_bts_{self.args.train_batchsize}_epo_{epoch}_F1_{round(100000*self.best_metric)}.pth"
            self.best_model_path = os.path.join(self.args.savepath, model_name)
            torch.save(state, self.best_model_path)
            print_log(f"Save Model => {self.best_model_path}", self.log)


if __name__ == "__main__":
    wandb.login()

    parser = argparse.ArgumentParser(description="Train Change3D")

    parser.add_argument(
        "--data_folder",
        default="./data/Forest-Change-dataset/images",
        help="folder with data files",
    )
    parser.add_argument(
        "--list_path", default="./data/Forest-Change/", help="path of the data lists"
    )
    parser.add_argument(
        "--token_folder",
        default="./data/Forest-Change/tokens/",
        help="folder with token files",
    )
    parser.add_argument(
        "--vocab_file",
        default="vocab",
        help="name of the vocab json (without extension)",
    )
    parser.add_argument(
        "--metadata_file",
        default="metadata",
        help="name of the metadata json (without extension)",
    )
    parser.add_argument(
        "--allow_unk", type=str2bool, default=True, help="if unknown token is allowed"
    )
    parser.add_argument(
        "--data_name", default="Forest-Change", help="dataset name identifier"
    )

    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id in the training.")
    parser.add_argument(
        "--print_freq",
        type=int,
        default=5,
        help="print training/validation stats every __ batches",
    )

    # Training parameters
    parser.add_argument(
        "--train_batchsize", type=int, default=32, help="batch_size for training"
    )
    parser.add_argument(
        "--augment",
        type=str2bool,
        default=False,
        help="whether to increase dataset size via augmentation or not",
    )
    parser.add_argument(
        "--increased_train_data_size",
        type=int,
        default=None,
        help="if provided, increases train dataset size to this value",
    )
    parser.add_argument(
        "--increased_val_data_size",
        type=int,
        default=None,
        help="if provided, increases validation dataset size to this value",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=120,
        help="number of epochs to train for (if early stopping is not triggered).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="number of epochs model doesn't improve by until training is ended early",
    )
    parser.add_argument("--workers", type=int, default=4, help="for data-loading")
    parser.add_argument(
        "--decoder_lr", type=float, default=2e-4, help="learning rate for optimization"
    )

    # Validation
    parser.add_argument(
        "--val_batchsize", type=int, default=1, help="batch_size for validation"
    )
    parser.add_argument("--savepath", default="./models_ckpt/")

    # Change3D-specific
    parser.add_argument(
        "--pretrained",
        default="model/X3D_L.pyth",
        type=str,
        help="Path to pretrained X3D",
    )
    parser.add_argument(
        "--num_perception_frame",
        type=int,
        default=1,
        help="Number of perception frames",
    )
    parser.add_argument("--num_classes", type=int, default=2)

    args = parser.parse_args()

    trainer = Change3DTrainer(args)

    print_log("\nStarting Epoch: {}".format(trainer.start_epoch), trainer.log)
    print_log("Total Epoches: {}".format(trainer.args.num_epochs), trainer.log)
    print_log(
        "Training Dataset Size: {}".format(trainer.train_dataset_size), trainer.log
    )
    try:
        for epoch in range(0, trainer.args.num_epochs):
            trainer.training(epoch)
            trainer.validation(epoch)
            if epoch - trainer.best_epoch > trainer.args.patience:
                print_log(
                    f"Model did not improve after {trainer.args.patience} epochs. Stopping training early.",
                    trainer.log,
                )
                break
    except Exception as e:
        print_log(f"Hit an exception: {e}", trainer.log)
                    f"Model did not improve after {trainer.args.patience} epochs. Stopping training early.",
                    trainer.log,
                )
                break
    except Exception as e:
        print_log(f"Hit an exception: {e}", trainer.log)
