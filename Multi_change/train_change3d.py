import argparse
import gc
import json
import os
import random
import time

import numpy as np
import wandb
from change3d.trainer import Trainer
from change3d.utils import BCEDiceLoss
from einops import rearrange
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from utils_tool.loss_funcs import *
from utils_tool.metrics import Evaluator
from utils_tool.utils import (
    accuracy,
    build_dataloaders,
    get_eval_score,
    print_log,
    str2bool,
    time_file_str,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISPLAY_PARAMS = False
NUM_TASKS = 2


class Change3DTrainer(object):
    def __init__(self, args):
        """
        Training and validation.
        """
        self.args = args
        self.run = wandb.init(
            project="forest-chat-change3d",
            config={
                "dataset_name": args.data_name,
                "train_batchsize": args.train_batchsize,
                "fine_tune_encoder": args.fine_tune_encoder,
                "val_batchsize": args.val_batchsize,
                "num_epochs": args.num_epochs,
                "patience": args.patience,
                "encoder_lr": args.encoder_lr,
                "decoder_cd_lr": args.cd_lr,
                "decoder_cc_lr": args.cc_lr,
                "num_classes": args.num_classes,
                "augment": args.augment,
                "num_perception_frame": args.num_perception_frame,
            },
        )
        random_str = str(random.randint(10, 100))
        name = (
            "change3d_"
            + time_file_str()
            + f"_loss_balancing_{args.loss_balancing_method}_"
            + f"_grad_method_{args.grad_method}_"
            + random_str
        )
        self.args.savepath = os.path.join(args.savepath, name)
        if os.path.exists(self.args.savepath) is False:
            os.makedirs(self.args.savepath)
        self.log = open(os.path.join(self.args.savepath, "{}.log".format(name)), "w")
        print_log(f"=>dataset: {args.data_name}", self.log)
        print_log(f"=>encoder_lr: {args.encoder_lr}", self.log)
        print_log(f"=>decoder_cd_lr: {args.cd_lr}", self.log)
        print_log(f"=>decoder_cc_lr: {args.cc_lr}", self.log)
        print_log(f"=>num_epochs: {args.num_epochs}", self.log)
        print_log(f"=>train_batchsize: {args.train_batchsize}", self.log)

        if args.loss_balancing_method == "uncert":
            self.log_vars = torch.nn.Parameter(torch.zeros(NUM_TASKS).to(DEVICE))

        if args.loss_balancing_method == "edwa":
            self.edwa = EDWA(num_epochs=args.num_epochs)

        self.best_bleu4 = 0.3  # BLEU-4 score right now
        self.MIou = 0.3
        self.Sum_Metric = 0.3
        self.start_epoch = 0

        with open(os.path.join(args.list_path + args.vocab_file + ".json"), "r") as f:
            self.word_vocab = json.load(f)

        args.vocab_size = len(self.word_vocab)
        self.beam_size = args.beam_size

        with open(
            os.path.join(args.list_path) + args.metadata_file + ".json", "r"
        ) as f:
            self.max_length = json.load(f)["max_length"]

        self.build_change3d_model()

        if args.grad_method != "none":
            self.rng = np.random.default_rng()
            self.grad_dims = []
            for param in self.model.parameters():
                self.grad_dims.append(param.data.numel())

            self.grads = torch.zeros(sum(self.grad_dims), NUM_TASKS).to(DEVICE)

        self.train_dataset_size, self.train_loader, self.val_loader = build_dataloaders(
            args, self.max_length
        )
        self.max_batches = len(self.train_loader)

        self.evaluator = Evaluator(num_class=args.num_classes)

        self.best_model_path = None
        self.best_epoch = 0

    def build_change3d_model(self):
        args = self.args

        self.model = Trainer(args).to(DEVICE)

        if args.load_from_checkpoint_and_train:
            if args.checkpoint is None:
                raise ValueError("Error: checkpoint is None.")

            checkpoint = torch.load(args.checkpoint)
            print(f"Load Model from {args.checkpoint}")

            self.start_epoch = checkpoint["epoch"]
            self.best_bleu4 = checkpoint["bleu-4"]
            self.MIou = checkpoint["MIoU"]
            self.model.encoder.load_state_dict(checkpoint["encoder_dict"])
            self.model.decoder_cd.load_state_dict(
                checkpoint["decoder_cd_dict"], strict=False
            )
            self.model.decoder_cc.load_state_dict(checkpoint["decoder_cc_dict"])

            self.model.encoder.eval()
            self.model.decoder_cd.eval()
            self.model.decoder_cc.eval()

        self.encoder_optimizer = None
        self.encoder_lr_scheduler = None
        if args.fine_tune_encoder:
            self.encoder_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.encoder.parameters()),
                lr=args.encoder_lr,
                weight_decay=1e-5,
            )
            self.encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.encoder_optimizer, step_size=900, gamma=1.0
            )

        self.cd_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.decoder_cd.parameters()),
            lr=args.cd_lr,
            weight_decay=1e-5,
        )
        self.cd_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.cd_optimizer, step_size=900, gamma=1.0
        )

        decoder_cc_params = list(
            filter(lambda p: p.requires_grad, self.model.decoder_cc.parameters())
        )
        if args.loss_balancing_method == "uncert":
            decoder_cc_params += [self.log_vars]

        self.cc_optimizer = torch.optim.Adam(
            decoder_cc_params,
            lr=args.cc_lr,
            weight_decay=1e-5,
        )
        self.cc_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.cc_optimizer, step_size=900, gamma=1.0
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)

        self.model = self.model.to(DEVICE)

        if DISPLAY_PARAMS:
            total_params = 0

            for name, param in self.model.named_parameters():
                print(
                    f"Layer: {name}\t Parameters: {param.numel()}\t Trainable: {param.requires_grad}"
                )
                total_params += param.numel()

            print(f"Total parameters: {total_params}")

    def training(self, args, epoch):
        # Only need one epoch of hist for printing as is stored in wandb to reduce memory
        self.index_i = 0
        self.hist = np.zeros((args.num_epochs * NUM_TASKS * len(self.train_loader), 5))

        self.model.encoder.train()
        self.model.decoder_cd.train()
        self.model.decoder_cc.train()

        accum_steps = 64 // args.train_batchsize

        for id, (imgA, imgB, seg_label, _, _, token, token_len, _) in enumerate(
            self.train_loader
        ):
            start_time = time.time()

            # Move to GPU, if available
            imgA = imgA.to(DEVICE)
            imgB = imgB.to(DEVICE)
            seg_label = seg_label.to(DEVICE)
            token = token.squeeze(1).to(DEVICE)
            token_len = token_len.to(DEVICE)

            seg_pred, percep_feat = self.model(imgA, imgB)
            percep_feat = rearrange(percep_feat, "b c h w -> (h w) b c")

            # Decode captions
            scores, caps_sorted, decode_lengths, sort_ind = self.model.decoder_cc(
                percep_feat, token, token_len
            )

            # Prepare targets and loss
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True
            ).data

            if seg_label.ndim == 3:
                seg_label = seg_label.unsqueeze(1)
            seg_label = seg_label.float()

            det_loss = BCEDiceLoss(seg_pred, seg_label)
            cap_loss = self.criterion(scores, targets)

            if args.loss_balancing_method == "uncert":
                loss = calc_uncertainty_weighting_loss(
                    det_loss, cap_loss, self.log_vars
                )
            elif args.loss_balancing_method == "edwa":
                loss = self.edwa.combine(det_loss, cap_loss, epoch)
            else:
                det_loss = det_loss / det_loss.detach().item()
                cap_loss = cap_loss / cap_loss.detach().item()
                loss = det_loss + cap_loss

            if self.args.grad_method != "none":
                (det_loss / accum_steps).backward(retain_graph=True)
                grad2vec(self.model, self.grads, self.grad_dims, 0)
                self.model.zero_grad()

                (cap_loss / accum_steps).backward(retain_graph=False)
                grad2vec(self.model, self.grads, self.grad_dims, 1)
                self.model.zero_grad()

                grad_methods = {
                    "cagrad": lambda: cagrad(self.grads, NUM_TASKS, 0.4, rescale=1),
                    "pcgrad": lambda: pcgrad(self.grads, self.rng, NUM_TASKS),
                    "graddrop": lambda: graddrop(self.grads),
                }

                try:
                    g = grad_methods[self.args.grad_method]()
                except KeyError as e:
                    raise ValueError(
                        f"Unknown grad_method: {self.args.grad_method}"
                    ) from e

                overwrite_grad(self.model, g, self.grad_dims, NUM_TASKS)

            else:
                (loss / accum_steps).backward()

            if args.grad_clip is not None:
                for params in self.model.parameters():
                    torch.nn.utils.clip_grad_value_(params, args.grad_clip)

            # Update weights
            if (id + 1) % accum_steps == 0 or (id + 1) == len(self.train_loader):
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.step()
                if self.cd_optimizer is not None:
                    self.cd_optimizer.step()
                if self.cc_optimizer is not None:
                    self.cc_optimizer.step()

                # Adjust learning rate
                if self.encoder_lr_scheduler is not None:
                    self.encoder_lr_scheduler.step()
                if self.cd_lr_scheduler is not None:
                    self.cd_lr_scheduler.step()
                if self.cc_lr_scheduler is not None:
                    self.cc_lr_scheduler.step()

                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.zero_grad()
                if self.cd_optimizer is not None:
                    self.cd_optimizer.zero_grad()
                if self.cc_optimizer is not None:
                    self.cc_optimizer.zero_grad()

            # Keep track of metrics
            self.hist[self.index_i, 0] = time.time() - start_time  # batch_time

            self.hist[self.index_i, 1] = det_loss.item()  # train_loss
            self.hist[self.index_i, 2] = accuracy(
                seg_pred.permute(0, 2, 3, 1).reshape(-1, seg_pred.size(1)),
                seg_label.reshape(-1),
                1,
            )

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
                    len(self.train_loader),
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
                    len(self.train_loader),
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
    @torch.no_grad()
    def validation(self, epoch):
        Caption_End = False

        self.model.encoder.eval()
        self.model.decoder_cd.eval()
        self.model.decoder_cc.eval()

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
                    desc="val_" + "EVALUATING AT BEAM SIZE " + str(self.beam_size),
                )
            ):
                k = self.beam_size
                # Move to GPU, if available
                imgA = imgA.to(DEVICE)
                imgB = imgB.to(DEVICE)
                token_all = token_all.squeeze(0).to(DEVICE)
                # Forward prop
                seg_pre, encoder_out = self.model(imgA, imgB)
                encoder_out = rearrange(encoder_out, "b c h w -> (h w) b c")

                # for segmentation
                pred_seg = seg_pre.data.cpu().numpy()
                seg_label = seg_label.cpu().numpy()
                pred_seg = np.argmax(pred_seg, axis=1)

                # Add batch sample into evaluator
                self.evaluator.add_batch(seg_label, pred_seg)

                # for captioning
                S, batch, encoder_dim = encoder_out.size()
                assert batch == 1, "Beam search only supports batch size 1."
                encoder_out = encoder_out.expand(S, k, encoder_dim).permute(1, 0, 2)

                tgt = torch.zeros(k, self.max_length, dtype=torch.int64, device=DEVICE)
                tgt[:, 0] = self.word_vocab["<START>"]
                seqs = torch.full(
                    (k, 1), self.word_vocab["<START>"], dtype=torch.int64, device=DEVICE
                )
                top_k_scores = torch.zeros(k, 1, device=DEVICE)

                complete_seqs = []
                complete_scores = []

                # causal mask
                mask = torch.triu(
                    torch.ones(self.max_length, self.max_length, device=DEVICE),
                    diagonal=1,
                )
                mask = mask.masked_fill(mask == 1, float("-inf")).masked_fill(
                    mask == 0, 0.0
                )

                for step in range(1, self.max_length):
                    # embedding + positional encoding
                    word_emb = self.model.decoder_cc.vocab_embedding(tgt[:, :step])
                    word_emb = word_emb.transpose(0, 1)
                    word_emb = self.model.decoder_cc.position_encoding(word_emb)

                    # transformer forward
                    enc = encoder_out.permute(1, 0, 2)
                    preds = self.model.decoder_cc.transformer(
                        word_emb, enc, tgt_mask=mask[:step, :step]
                    )
                    preds = self.model.decoder_cc.wdc(preds)
                    scores = F.log_softmax(preds[-1], dim=-1)
                    scores = (
                        top_k_scores.expand_as(scores) + scores
                    )  # add accumulated scores

                    # select top k
                    if step == 1:
                        top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
                    else:
                        top_k_scores, top_k_words = scores.view(-1).topk(
                            k, 0, True, True
                        )

                    prev_word_inds = torch.div(
                        top_k_words, args.vocab_size, rounding_mode="floor"
                    )
                    next_word_inds = top_k_words % args.vocab_size

                    # build sequences
                    seqs = torch.cat(
                        [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1
                    )

                    # find completed and incomplete
                    incomplete_inds = [
                        i
                        for i, w in enumerate(next_word_inds)
                        if w != self.word_vocab["<END>"]
                    ]
                    complete_inds = list(
                        set(range(len(next_word_inds))) - set(incomplete_inds)
                    )

                    if len(complete_inds) > 0:
                        complete_seqs.extend(seqs[complete_inds].tolist())
                        complete_scores.extend(top_k_scores[complete_inds].tolist())

                    k -= len(complete_inds)
                    if k == 0:
                        break

                    seqs = seqs[incomplete_inds]
                    encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                    top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                    tgt = tgt[incomplete_inds]
                    tgt[:, : step + 1] = seqs

                # fallback if no completed sequences
                if len(complete_seqs) == 0:
                    complete_seqs = seqs.tolist()
                    complete_scores = top_k_scores.squeeze(1).tolist()

                # pick best sequence
                i = complete_scores.index(max(complete_scores))
                seq = complete_seqs[i]

                # --- reference and hypothesis lists ---
                img_caps = token_all.tolist()
                img_captions = [
                    [
                        w
                        for w in c
                        if w
                        not in {
                            self.word_vocab["<START>"],
                            self.word_vocab["<END>"],
                            self.word_vocab["<NULL>"],
                        }
                    ]
                    for c in img_caps
                ]
                references.append(img_captions)

                hyp = [
                    w
                    for w in seq
                    if w
                    not in {
                        self.word_vocab["<START>"],
                        self.word_vocab["<END>"],
                        self.word_vocab["<NULL>"],
                    }
                ]
                hypotheses.append(hyp)
                assert len(references) == len(hypotheses)

                # if ind % self.args.print_freq == 0:
                #     pred_caption = " ".join(
                #         [list(self.word_vocab.keys())[i] for i in hyp]
                #     )
                #     print(f"[{ind}] Pred: {pred_caption}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            val_time = time.time() - val_start_time

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
        Sum_Metric = mIoU_seg + Bleu_4
        if Sum_Metric >= self.Sum_Metric:
            self.best_bleu4 = max(Bleu_4, self.best_bleu4)
            self.MIou = max(mIoU_seg, self.MIou)
            self.Sum_Metric = max(Sum_Metric, self.Sum_Metric)

            state = {
                "epoch": epoch,
                "bleu-4": self.best_bleu4,
                "MIoU": self.MIou,
                "encoder_dict": self.model.encoder.state_dict(),
                "decoder_cd_dict": self.model.decoder_cd.state_dict(),
                "decoder_cc_dict": self.model.decoder_cc.state_dict(),
                "encoder_optimizer": self.encoder_optimizer,
                "decoder_cd_optimizer": self.cd_optimizer,
                "decoder_cc_optimizer": self.cc_optimizer,
            }
            metric = f"Sum_{round(100000*self.Sum_Metric)}_MIou_{round(100000*self.MIou)}_Bleu4_{round(100000*self.best_bleu4)}"
            # metric = f'MIou_{round(10000 * self.MIou)}_Bleu4_{round(10000 * self.best_bleu4)}'
            model_name = f"{self.args.data_name}_bts_{self.args.train_batchsize}_epo_{epoch}_{metric}.pth"
            best_model_path = os.path.join(self.args.savepath, model_name)

            if epoch > 10:
                # save_checkpoint
                print("Save Model")
                torch.save(state, best_model_path)
            self.best_epoch = epoch
            self.best_model_path = best_model_path


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
        "--checkpoint",
        default=None,
        help="path to checkpoint",
    )
    parser.add_argument(
        "--pretrained",
        default="models_ckpt/X3D_L.pyth",
        type=str,
        help="Path to pretrained weight",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=5,
        help="print training/validation stats every __ batches",
    )
    # Training parameters
    parser.add_argument(
        "--load_from_checkpoint_and_train",
        type=str2bool,
        default=False,
        help="whether to load a checkpoint and continue training from it",
    )
    parser.add_argument(
        "--fine_tune_encoder",
        type=str2bool,
        default=True,
        help="whether fine-tune encoder or not",
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
        default=100,
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
        "--encoder_lr",
        type=float,
        default=1e-4,
        help="Learning rate for encoder if fine-tuning.",
    )
    parser.add_argument(
        "--cd_lr",
        type=float,
        default=1e-4,
        help="Learning rate for cd decoder.",
    )
    parser.add_argument(
        "--cc_lr",
        type=float,
        default=1e-4,
        help="Learning rate for cc decoder.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="clip gradients at an absolute value of.",
    )
    # Validation
    parser.add_argument(
        "--val_batchsize", type=int, default=1, help="batch_size for validation"
    )
    parser.add_argument("--savepath", default="./models_ckpt/")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument(
        "--loss_balancing_method",
        default="uncert",
        help="loss balancing approach method",
        choices=["edwa", "uncert", "normalised"],
    )
    parser.add_argument(
        "--grad_method",
        default="none",
        help="gradient adjustment method for resolving potential conflicts between multiple learning tasks",
        choices=["none", "pcgrad", "graddrop", "cagrad"],
    )
    parser.add_argument(
        "--n_head", type=int, default=8, help="Multi-head attention in Transformer."
    )
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--decoder_n_layers", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--num_perception_frame",
        type=int,
        default=1,
        help="Number of perception frames",
    )
    parser.add_argument(
        "--beam_size", type=int, default=1, help="Beam size for beam search."
    )
    parser.add_argument(
        "--in_height", type=int, default=256, help="Height of RGB image"
    )
    parser.add_argument("--in_width", type=int, default=256, help="Width of RGB image")
    args = parser.parse_args()

    trainer = Change3DTrainer(args)
    print_log("\nStarting Epoch: {}".format(trainer.start_epoch), trainer.log)
    print_log("Total Epoches: {}".format(trainer.args.num_epochs), trainer.log)
    print_log(
        "Training Dataset Size: {}".format(trainer.train_dataset_size), trainer.log
    )

    try:
        for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
            trainer.training(trainer.args, epoch)
            trainer.validation(epoch)
            if epoch - trainer.best_epoch > trainer.args.patience:
                trainer.start_epoch = trainer.best_epoch + 1
                trainer.args.num_epochs = trainer.start_epoch + trainer.args.num_epochs
                print_log(
                    f"Model did not improve after {trainer.args.patience} epochs. Stopping training early.",
                    trainer.log,
                )
                break
    except Exception as e:
        print_log("Hit an exception: {}".format(e), trainer.log)
