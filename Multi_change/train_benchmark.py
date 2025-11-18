import argparse
import gc
import json
import os
import random
import time
from distutils.util import strtobool
from pathlib import Path

import numpy as np
import wandb
from benchmark_models.change_3d.trainer import Change3d_Trainer
from benchmark_models.change_3d.utils import BCEDiceLoss, adjust_lr
from data.ForestChange import ForestChangeDataset
from data.LEVIR_MCI import LEVIRCCDataset
from einops import rearrange
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from tqdm import tqdm
from utils_tool.loss_funcs import *
from utils_tool.metrics import Evaluator
from utils_tool.utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(self, args):
        """
        Training and validation.
        """
        self.start_train_goal = args.train_goal
        self.args = args
        self.run = wandb.init(
            project="forest-chat",
            config={
                "dataset_name": args.data_name,
                "train_goal": args.train_goal,
                "train_batchsize": args.train_batchsize,
                "num_epochs": args.num_epochs,
                "patience": args.patience,
                "benchmark_model": args.benchmark,
            },
        )
        random_str = str(random.randint(10, 100))
        name = (
            f"{args.benchmark}_"
            + time_file_str()
            + f"_train_goal_{args.train_goal}_"
            + random_str
        )
        self.args.savepath = os.path.join(args.savepath, name)

        if os.path.exists(self.args.savepath) == False:
            os.makedirs(self.args.savepath)
        self.log = open(os.path.join(self.args.savepath, "{}.log".format(name)), "w")
        print_log("=>device: {}".format(DEVICE), self.log)
        print_log("=>dataset: {}".format(args.data_name), self.log)
        print_log("=>num_epochs: {}".format(args.num_epochs), self.log)
        print_log("=>train_batchsize: {}".format(args.train_batchsize), self.log)
        print_log("=>benchmark: {}".format(args.benchmark), self.log)

        self.best_bleu4 = 0.3  # BLEU-4 score right now
        self.MIou = 0.3
        self.start_epoch = 0
        with open(os.path.join(args.list_path + args.vocab_file + ".json"), "r") as f:
            args.word_vocab = json.load(f)

        with open(
            os.path.join(args.list_path) + args.metadata_file + ".json", "r"
        ) as f:
            args.max_length = json.load(f)["max_length"]

        args.vocab_size = len(args.word_vocab)
        # Initialize / load checkpoint
        self.build_benchmark_model()

        # Custom dataloaders
        if args.data_name in ["LEVIR_MCI", "Forest-Change"]:
            datasets = []
            for split in ["train", "val"]:
                dataset = (
                    ForestChangeDataset(
                        data_folder=args.data_folder,
                        list_path=args.list_path,
                        split=split,
                        token_folder=args.token_folder,
                        vocab_file=args.vocab_file,
                        max_length=args.max_length,
                        allow_unk=args.allow_unk,
                        transform=get_image_transforms() if args.augment else None,
                        max_iters=(
                            args.increased_train_data_size
                            if split == "train"
                            else args.increased_val_data_size
                        ),
                        num_classes=args.num_class,
                    )
                    if "Forest-Change" in args.data_name
                    else LEVIRCCDataset(
                        data_folder=args.data_folder,
                        list_path=args.list_path,
                        split=split,
                        token_folder=args.token_folder,
                        vocab_file=args.vocab_file,
                        max_length=args.max_length,
                        allow_unk=args.allow_unk,
                        num_classes=args.num_class,
                    )
                )
                datasets.append(dataset)
            self.train_dataset_size = len(datasets[0])
            self.train_loader = data.DataLoader(
                datasets[0],
                batch_size=args.train_batchsize,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
            )
            self.max_batches = len(self.train_loader)
            self.val_loader = data.DataLoader(
                datasets[1],
                batch_size=args.val_batchsize,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
            )

        self.evaluator = Evaluator(num_class=args.num_class)

        self.best_model_path = None
        self.best_epoch = 0

    def build_benchmark_model(self):
        args = self.args

        if args.benchmark == "change_3d":
            self.model = Change3d_Trainer(args).to(DEVICE).float()
            if args.train_goal == 0:
                self.optimiser = torch.optim.Adam(
                    self.model.parameters(),
                    args.lr,
                    (0.9, 0.99),
                    eps=1e-08,
                    weight_decay=1e-4,
                )
            elif args.train_goal == 1:
                self.encoder_optimiser = None
                self.encoder_lr_scheduler = None

                if args.fine_tune_encoder:
                    self.encoder_optimiser = torch.optim.Adam(
                        params=filter(
                            lambda p: p.requires_grad, self.model.encoder.parameters()
                        ),
                        lr=args.encoder_lr,
                        weight_decay=1e-5,
                    )
                    self.encoder_lr_scheduler = StepLR(
                        self.encoder_optimiser, step_size=900, gamma=1
                    )

                self.decoder_optimiser = torch.optim.Adam(
                    params=filter(
                        lambda p: p.requires_grad, self.model.decoder.parameters()
                    ),
                    lr=args.decoder_lr,
                    weight_decay=1e-5,
                )
                self.decoder_lr_scheduler = StepLR(
                    self.decoder_optimiser, step_size=900, gamma=1
                )
                self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)
            else:
                raise ValueError("Unknown train goal selected.")
        else:
            raise ValueError("Unknown benchmark model selected.")

        # --- Load checkpoint if resuming ---
        if args.checkpoint is not None:
            print_log(f"Resuming from checkpoint: {args.checkpoint}", self.log)
            checkpoint = torch.load(args.checkpoint, map_location=DEVICE)

            self.model.load_state_dict(checkpoint["state_dict"])

            self.start_epoch = checkpoint.get("epoch", 0)
            self.best_epoch = checkpoint.get("epoch", 0)
            self.MIoU = checkpoint.get("best_mIoU", 0.3)
            self.best_bleu4 = checkpoint.get("best_bleu4", 0.3)

    def training(self, args, epoch, lr_factor=1.0):
        # Only need one epoch of hist for printing as is stored in wandb to reduce memory
        self.index_i = 0
        self.hist = np.zeros((args.num_epochs * self.max_batches, 5))

        if args.benchmark == "change_3d":
            if args.train_goal == 1:
                self.model.encoder.train()
                self.model.decoder.train()
                if epoch > 0 and epoch % 10 == 0:
                    adjust_lr(
                        args=None,
                        optimizer=self.encoder_optimiser,
                        shrink_factor=0.5,
                    )
                    adjust_lr(
                        args=None,
                        optimizer=self.decoder_optimiser,
                        shrink_factor=0.5,
                    )
            else:
                self.model.train()

        for id, (imgA, imgB, seg_label, _, _, token, token_len, _) in enumerate(
            self.train_loader
        ):
            start_time = time.time()

            if args.data_name == "LEVIR_MCI":
                seg_label = (seg_label > 0).long()
                args.num_class = 2  # enforce

            imgA = imgA.to(DEVICE)
            imgB = imgB.to(DEVICE)
            seg_label = seg_label.to(DEVICE)

            token = token.squeeze(1).to(DEVICE)
            token_len = token_len.to(DEVICE)

            if args.benchmark == "change_3d":
                if args.train_goal == 1:
                    det_loss = torch.tensor(0.0, device=DEVICE)

                    percep_feat = self.model.update_cc(imgA, imgB)
                    percep_feat = rearrange(percep_feat, "b c h w -> (h w) b c")

                    scores, caps_sorted, decode_lengths, sort_ind = self.model.decoder(
                        percep_feat, token, token_len
                    )
                    targets = caps_sorted[:, 1:]

                    scores = pack_padded_sequence(
                        scores, decode_lengths, batch_first=True
                    ).data
                    targets = pack_padded_sequence(
                        targets, decode_lengths, batch_first=True
                    ).data

                    cap_loss = self.criterion(scores, targets)

                    self.decoder_optimiser.zero_grad()
                    if self.encoder_optimiser is not None:
                        self.encoder_optimiser.zero_grad()
                    cap_loss.backward()

                    if args.grad_clip is not None:
                        clip_gradient(self.decoder_optimiser, args.grad_clip)
                        if self.encoder_optimiser is not None:
                            clip_gradient(self.encoder_optimiser, args.grad_clip)

                    # Before backward
                    print(
                        f"LR - Encoder: {self.encoder_optimiser.param_groups[0]['lr']:.2e}"
                    )
                    print(
                        f"LR - Decoder: {self.decoder_optimiser.param_groups[0]['lr']:.2e}"
                    )

                    # After backward, before step
                    encoder_grad_norm = torch.norm(
                        torch.stack(
                            [
                                p.grad.norm()
                                for p in self.model.encoder.parameters()
                                if p.grad is not None
                            ]
                        )
                    )
                    decoder_grad_norm = torch.norm(
                        torch.stack(
                            [
                                p.grad.norm()
                                for p in self.model.decoder.parameters()
                                if p.grad is not None
                            ]
                        )
                    )
                    print(
                        f"Gradient norms - Encoder: {encoder_grad_norm:.4f}, Decoder: {decoder_grad_norm:.4f}"
                    )

                    # Check for dead neurons
                    dead_encoder = sum(
                        (p == 0).float().mean() > 0.9
                        for p in self.model.encoder.parameters()
                    )
                    dead_decoder = sum(
                        (p == 0).float().mean() > 0.9
                        for p in self.model.decoder.parameters()
                    )
                    print(
                        f"Dead param check - Encoder: {dead_encoder}, Decoder: {dead_decoder}"
                    )

                    # Update weights
                    self.encoder_optimiser.step()
                    self.encoder_lr_scheduler.step()
                    self.decoder_optimiser.step()
                    self.decoder_lr_scheduler.step()
                else:
                    cap_loss = torch.tensor(0.0, device=DEVICE)
                    if seg_label.ndim == 3:
                        seg_label = seg_label.unsqueeze(1)
                    seg_label = seg_label.float()

                    adjust_lr(
                        args,
                        self.optimiser,
                        epoch,
                        id + self.index_i,
                        self.max_batches,
                        lr_factor=lr_factor,
                    )
                    seg_pred = self.model.update_bcd(imgA, imgB)
                    det_loss = BCEDiceLoss(seg_pred, seg_label)

                    seg_pred = torch.where(
                        seg_pred > 0.5,
                        torch.ones_like(seg_pred),
                        torch.zeros_like(seg_pred),
                    ).long()
                    self.optimiser.zero_grad()
                    det_loss.backward()
                    self.optimiser.step()
            else:
                pass

            # Keep track of metrics
            self.hist[self.index_i, 0] = time.time() - start_time  # batch_time
            if self.args.train_goal == 0:
                self.hist[self.index_i, 1] = det_loss.item()  # train_loss
                self.hist[self.index_i, 2] = accuracy(
                    seg_pred.permute(0, 2, 3, 1).reshape(-1, seg_pred.size(1)),
                    seg_label.reshape(-1),
                    1,
                )
            if self.args.train_goal == 1:
                self.hist[self.index_i, 3] = cap_loss.item()  # train_loss
                self.hist[self.index_i, 4] = accuracy(scores, targets, 5)  # top5

            self.index_i += 1
            log_vals = False
            # Print status
            if self.index_i % args.print_freq == 0 and args.print_freq > 1:
                log_vals = True
                print_vals = (
                    epoch,
                    id,
                    self.max_batches,
                    np.mean(
                        self.hist[self.index_i - args.print_freq : self.index_i - 1, 0]
                    )
                    * args.print_freq,
                    np.mean(
                        self.hist[self.index_i - args.print_freq : self.index_i - 1, 1]
                    ),
                    np.mean(
                        self.hist[self.index_i - args.print_freq : self.index_i - 1, 2]
                    ),
                    np.mean(
                        self.hist[self.index_i - args.print_freq : self.index_i - 1, 3]
                    ),
                    np.mean(
                        self.hist[self.index_i - args.print_freq : self.index_i - 1, 4]
                    ),
                )
            elif args.print_freq == 1:
                log_vals = True
                print_vals = (
                    epoch,
                    id,
                    self.max_batches,
                    self.hist[self.index_i - 1, 0],
                    self.hist[self.index_i - 1, 1],
                    self.hist[self.index_i - 1, 2],
                    self.hist[self.index_i - 1, 3],
                    self.hist[self.index_i - 1, 4],
                )

            if log_vals:
                print_log(
                    "Training Epoch: [{0}][{1}/{2}]\t"
                    "Batch Time: {3:.3f}\t"
                    "Det_Loss: {4:.4f}\t"
                    "Det Acc: {5:.3f}\t"
                    "Cap_Loss: {6:.5f}\t"
                    "Text_Top-5 Acc: {7:.3f}".format(
                        print_vals[0],
                        print_vals[1],
                        print_vals[2],
                        print_vals[3],
                        print_vals[4],
                        print_vals[5],
                        print_vals[6],
                        print_vals[7],
                    ),
                    self.log,
                )
                wandb.log(
                    {
                        "epoch": print_vals[0],
                        "batch_id": print_vals[1],
                        "detection_train_loss": print_vals[4],
                        "detection_train_accuracy": print_vals[5],
                        "caption_train_loss": print_vals[6],
                        "text_top_5_train_accuracy": print_vals[7],
                    }
                )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # One epoch's validation
    def validation(self, epoch):
        word_vocab = self.args.word_vocab

        if args.benchmark == "change_3d":
            if args.train_goal == 1:
                self.model.encoder.to(DEVICE).eval()
                self.model.decoder.to(DEVICE).eval()
            else:
                self.model.to(DEVICE).eval()

        val_start_time = time.time()
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        self.evaluator.reset()

        with torch.no_grad():
            # Batches
            for ind, (
                imgA,
                imgB,
                seg_label,
                token_all,
                token_all_len,
                _,
                _,
                _,
            ) in enumerate(
                tqdm(
                    self.val_loader,
                    desc="val_" + "EVALUATING AT BEAM SIZE " + str(args.beam_size),
                )
            ):
                if args.data_name == "LEVIR_MCI":
                    seg_label = (seg_label > 0).long()
                    args.num_class = 2  # enforce

                # Move to GPU, if available
                imgA = imgA.to(DEVICE)
                imgB = imgB.to(DEVICE)
                seg_label = seg_label.to(DEVICE)
                token_all = token_all.squeeze(0).to(DEVICE)

                if args.benchmark == "change_3d":
                    if args.train_goal == 1:
                        encoder_out = self.model.update_cc(imgA, imgB)
                        encoder_out = rearrange(encoder_out, "b c h w -> (h w) b c")

                        seq = self.model.decoder.sample_beam(
                            encoder_out, k=args.beam_size
                        )

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

                        if ind % self.args.print_freq == 0:
                            pred_caption = ""
                            ref_caption = ""
                            for i in pred_seq:
                                pred_caption += (list(word_vocab.keys())[i]) + " "
                            ref_caption = ""
                            for i in img_tokens:
                                for j in i:
                                    ref_caption += (list(word_vocab.keys())[j]) + " "
                                ref_caption += ".    "
                    else:
                        seg_pred = self.model.update_bcd(imgA, imgB)
                        seg_pred = seg_pred.squeeze(1)
                        seg_pred = torch.where(
                            seg_pred > 0.5,
                            torch.ones_like(seg_pred),
                            torch.zeros_like(seg_pred),
                        ).long()
                        pred_seg = seg_pred.data.cpu().numpy()
                        seg_label = seg_label.cpu().numpy()

                        self.evaluator.add_batch(seg_label, pred_seg)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            val_time = time.time() - val_start_time
            # Fast test during the training
            # for segmentation
            if self.args.train_goal == 0:
                Acc_seg = self.evaluator.Pixel_Accuracy()
                Acc_class_seg = self.evaluator.Pixel_Accuracy_Class()
                mIoU_seg, IoU = self.evaluator.Mean_Intersection_over_Union()
                FWIoU_seg = self.evaluator.Frequency_Weighted_Intersection_over_Union()
                Acc_seg = self.evaluator.Pixel_Accuracy()
                F1_score, F1_class_score = self.evaluator.F1_Score()
                print(
                    "Validation:\n"
                    "Acc_seg: {0:.5f}\t"
                    "Acc_class_seg: {1:.5f}\t"
                    "mIoU_seg: {2:.5f}\t"
                    "FWIoU_seg: {3:.5f}\t"
                    "IoU: {4}\t"
                    "F1: {5:.5f}\t"
                    "F1_class: {6}\t".format(
                        Acc_seg,
                        Acc_class_seg,
                        mIoU_seg,
                        FWIoU_seg,
                        IoU,
                        F1_score,
                        F1_class_score,
                    )
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "acc_seg_val": Acc_seg,
                        "acc_class_seg_val": Acc_class_seg,
                        "mIoU seg val": mIoU_seg,
                        "FWIoU seg": FWIoU_seg,
                        "IoU": IoU,
                        "F1": F1_score,
                    }
                )

            # Calculate evaluation scores
            if self.args.train_goal == 1:
                score_dict = get_eval_score(references, hypotheses)
                Bleu_1 = score_dict["Bleu_1"]
                Bleu_2 = score_dict["Bleu_2"]
                Bleu_3 = score_dict["Bleu_3"]
                Bleu_4 = score_dict["Bleu_4"]
                Meteor = score_dict["METEOR"]
                Rouge = score_dict["ROUGE_L"]
                Cider = score_dict["CIDEr"]
                print_log(
                    "Captioning_Validation:\n"
                    "Epoch: {0}\t"
                    "Time: {1:.3f}\t"
                    "BLEU-1: {2:.5f}\t"
                    "BLEU-2: {3:.5f}\t"
                    "BLEU-3: {4:.5f}\t"
                    "BLEU-4: {5:.5f}\t"
                    "Meteor: {6:.5f}\t"
                    "Rouge: {7:.5f}\t"
                    "Cider: {8:.5f}\t".format(
                        epoch,
                        val_time,
                        Bleu_1,
                        Bleu_2,
                        Bleu_3,
                        Bleu_4,
                        Meteor,
                        Rouge,
                        Cider,
                    ),
                    self.log,
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "BLEU-1_val": Bleu_1,
                        "BLEU-2_val": Bleu_2,
                        "BLEU-3_val": Bleu_3,
                        "BLEU-4_val": Bleu_4,
                        "Meteor_val": Meteor,
                        "Rouge_val": Rouge,
                        "Cider": Cider,
                    }
                )

        # Check if there was an improvement
        if args.train_goal == 0:
            Bleu_4 = 0
        if args.train_goal == 1:
            mIoU_seg = 0
        if Bleu_4 > self.best_bleu4 or mIoU_seg > self.MIou:
            self.best_bleu4 = max(Bleu_4, self.best_bleu4)
            self.MIou = max(mIoU_seg, self.MIou)

            if args.train_goal == 0:
                state = {
                    "state_dict": self.model.state_dict(),
                    "arch": str(self.model),
                    "epoch": epoch + 1,
                    "best_mIoU": self.MIou,
                    "optimiser": self.optimiser.state_dict(),
                }
            else:
                state = {
                    "epoch": epoch + 1,
                    "arch": str(args.benchmark),
                    "best_bleu4": self.best_bleu4,
                    "encoder_state_dict": self.model.encoder.state_dict(),
                    "decoder_state_dict": self.model.decoder.state_dict(),
                    "encoder_image_optimizer": self.encoder_optimiser.state_dict(),
                    "decoder_optimizer": self.decoder_optimiser.state_dict(),
                }

            # save_checkpoint
            metric = f"MIou_{round(100000 * self.MIou)}_Bleu4_{round(100000 * self.best_bleu4)}"
            model_name = f"{args.benchmark}_{args.data_name}_bts_{args.train_batchsize}_epo_{epoch}_{metric}.pth"
            best_model_path = os.path.join(self.args.savepath, model_name)

            if epoch > 5:
                print(f"Save Model: {best_model_path}")
                torch.save(state, os.path.join(args.savepath, model_name))

            self.best_epoch = epoch
            self.best_model_path = best_model_path


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
        "--allow_unk", type=str2bool, default=True, help="if unknown token is allowed"
    )
    parser.add_argument(
        "--data_name", default="Forest-Change", help="base name shared by data files."
    )

    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id in the training.")

    parser.add_argument(
        "--benchmark",
        default=None,
        help="name of the benchmark model to be loaded",
        choices=["change_3d"],
    )

    parser.add_argument(
        "--print_freq",
        type=int,
        default=5,
        help="print training/validation stats every __ batches",
    )
    # Training parameters
    parser.add_argument(
        "--train_goal",
        type=int,
        default=0,
        help="0:det; 1:cap;",
        choices=[0, 1],
    )
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
        help="if you provide a number, it will increase the train dataset size to match the number",
    )
    parser.add_argument(
        "--increased_val_data_size",
        type=int,
        default=None,
        help="if you provide a number, it will increase the validation dataset size to match the number",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=250,
        help="number of epochs to train for (if early stopping is not triggered).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="number of epochs model doesn't improve by until training is ended early",
    )
    parser.add_argument("--workers", type=int, default=4, help="for data-loading")

    # Validation
    parser.add_argument(
        "--val_batchsize", type=int, default=1, help="batch_size for validation"
    )
    parser.add_argument("--savepath", default="./models_ckpt/")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="path to checkpoint for resuming training",
    )
    parser.add_argument("--num_class", type=int, default=2)

    args = parser.parse_args()

    json_params = Path(
        f"./benchmark_models/{args.benchmark}/parameters.json"
    ).read_text()
    json_args = json.loads(json_params)

    for k, v in json_args.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)

    trainer = Trainer(args)
    print_log("\nStarting Epoch: {}".format(trainer.start_epoch), trainer.log)
    print_log("Total Epochs: {}".format(trainer.args.num_epochs), trainer.log)
    print_log(
        "Training Dataset Size: {}".format(trainer.train_dataset_size), trainer.log
    )

    try:
        for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
            trainer.training(trainer.args, epoch)
            # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
            if epoch - trainer.best_epoch > trainer.args.patience:
                print_log(
                    f"Model did not improve after {trainer.args.patience} epochs. Stopping training early.",
                    trainer.log,
                )
                break
    except Exception as e:
        print_log("Hit an exception: {}".format(e), trainer.log)
