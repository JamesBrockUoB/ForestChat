import argparse
import gc
import json
import random
import time

import numpy as np
import torch.optim
import wandb
from data.ForestChange import ForestChangeDataset
from data.LEVIR_MCI import LEVIRCCDataset
from model.model_decoder import DecoderTransformer
from model.model_encoder_att import AttentiveEncoder, Encoder
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
from tqdm import tqdm
from utils_tool.metrics import Evaluator
from utils_tool.utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASS = 2  # 3 for LEVIR-MCI
DISPLAY_PARAMS = False


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
                "train_state": args.train_stage,
                "fine_tune_encoder": args.fine_tune_encoder,
                "train_batchsize": args.train_batchsize,
                "num_epochs": args.num_epochs,
                "patience": args.patience,
                "encoder_lr": args.encoder_lr,
                "decoder_lr": args.decoder_lr,
                "network": args.network,
                "encoder_dim": args.encoder_dim,
                "feat_size": args.feat_size,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "decoder_n_layers": args.decoder_n_layers,
                "feature_dim": args.feature_dim,
            },
        )
        random_str = str(random.randint(10, 100))
        name = (
            "baseline_"
            + time_file_str()
            + f"_train_goal_{args.train_goal}_"
            + random_str
        )
        self.args.savepath = os.path.join(args.savepath, name)
        self.args.savepath = os.path.join(args.savepath, name)
        if os.path.exists(self.args.savepath) == False:
            os.makedirs(self.args.savepath)
        self.log = open(os.path.join(self.args.savepath, "{}.log".format(name)), "w")
        print_log("=>dataset: {}".format(args.data_name), self.log)
        print_log("=>network: {}".format(args.network), self.log)
        print_log("=>encoder_lr: {}".format(args.encoder_lr), self.log)
        print_log("=>decoder_lr: {}".format(args.decoder_lr), self.log)
        print_log("=>num_epochs: {}".format(args.num_epochs), self.log)
        print_log("=>train_batchsize: {}".format(args.train_batchsize), self.log)

        self.best_bleu4 = 0.4  # BLEU-4 score right now
        self.MIou = 0.4
        self.Sum_Metric = 0.4
        self.start_epoch = 0
        with open(os.path.join(args.list_path + args.vocab_file + ".json"), "r") as f:
            self.word_vocab = json.load(f)

        with open(
            os.path.join(args.list_path) + args.metadata_file + ".json", "r"
        ) as f:
            self.max_length = json.load(f)["max_length"]
        # Initialize / load checkpoint
        self.build_model()

        # Loss function
        self.criterion_cap = torch.nn.CrossEntropyLoss().to(DEVICE)
        self.criterion_det = torch.nn.CrossEntropyLoss().to(DEVICE)

        # Custom dataloaders
        if args.data_name in ["LEVIR_MCI", "Forest-Change"]:
            datasets = []
            for split in ["train", "val"]:
                dataset = (
                    ForestChangeDataset(
                        args.data_folder,
                        args.list_path,
                        split,
                        args.token_folder,
                        args.vocab_file,
                        self.max_length,
                        args.allow_unk,
                        get_image_transforms(),
                        (
                            args.increased_train_data_size
                            if split == "train"
                            else args.increased_val_data_size
                        ),
                    )
                    if "Forest-Change" in args.data_name
                    else LEVIRCCDataset(
                        args.data_folder,
                        args.list_path,
                        split,
                        args.token_folder,
                        args.vocab_file,
                        self.max_length,
                        args.allow_unk,
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
            self.val_loader = data.DataLoader(
                datasets[1],
                batch_size=args.val_batchsize,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
            )
        # Epochs

        self.evaluator = Evaluator(num_class=NUM_CLASS)

        self.best_model_path = None
        self.best_epoch = 0

    def debug_gpu_memory(self, tag=""):
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print_log(
            f"[GPU MEM] {tag} | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB",
            self.log,
        )

    def list_gpu_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    print_log(
                        f"Tensor: {tuple(obj.size())}, dtype={obj.dtype}, requires_grad={obj.requires_grad}",
                        self.log,
                    )
            except Exception:
                pass

    def build_model(self):
        args = self.args
        if args.train_stage == "s1":
            self.encoder = Encoder(args.network)
            self.encoder.fine_tune(args.fine_tune_encoder)
            self.encoder_trans = AttentiveEncoder(
                train_stage=args.train_stage,
                n_layers=args.n_layers,
                feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                heads=args.n_heads,
                dropout=args.dropout,
            )
            self.decoder = DecoderTransformer(
                encoder_dim=args.encoder_dim,
                feature_dim=args.feature_dim,
                vocab_size=len(self.word_vocab),
                max_lengths=self.max_length,
                word_vocab=self.word_vocab,
                n_head=args.n_heads,
                n_layers=args.decoder_n_layers,
                dropout=args.dropout,
            )

            fine_tune_capdecoder = True
        elif args.train_stage == "s2" and args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint)
            print("Load Model from {}".format(args.checkpoint))
            # self.start_epoch = checkpoint['epoch'] + 1
            # self.best_bleu4 = checkpoint['bleu-4']
            self.decoder.load_state_dict(checkpoint["decoder_dict"])
            self.encoder_trans.load_state_dict(
                checkpoint["encoder_trans_dict"], strict=False
            )
            self.encoder.load_state_dict(checkpoint["encoder_dict"])
            # eval()
            self.encoder.eval()
            self.encoder_trans.eval()
            self.decoder.eval()
            # 各个modules 是否需要微调
            args.fine_tune_encoder = False
            self.encoder.fine_tune(args.fine_tune_encoder)
            self.encoder_trans.fine_tune(args.train_goal)
            fine_tune_capdecoder = False if args.train_goal == 0 else True
            self.decoder.fine_tune(fine_tune_capdecoder)
        else:
            # print('Error: checkpoint is None or stage=s1.')
            raise ValueError("Error: checkpoint is None.")

        # set optimizer
        self.encoder_optimizer = (
            torch.optim.Adam(params=self.encoder.parameters(), lr=args.encoder_lr)
            if args.fine_tune_encoder
            else None
        )
        self.encoder_trans_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.encoder_trans.parameters()),
            lr=args.encoder_lr,
        )
        self.decoder_optimizer = (
            torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
                lr=args.decoder_lr,
            )
            if fine_tune_capdecoder
            else None
        )

        # Move to GPU, if available
        self.encoder = self.encoder.to(DEVICE)
        self.encoder_trans = self.encoder_trans.to(DEVICE)
        self.decoder = self.decoder.to(DEVICE)
        self.encoder_lr_scheduler = (
            torch.optim.lr_scheduler.StepLR(
                self.encoder_optimizer, step_size=5, gamma=1.0
            )
            if args.fine_tune_encoder
            else None
        )
        self.encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.encoder_trans_optimizer, step_size=5, gamma=1.0
        )
        self.decoder_lr_scheduler = (
            torch.optim.lr_scheduler.StepLR(
                self.decoder_optimizer, step_size=5, gamma=1.0
            )
            if fine_tune_capdecoder
            else None
        )

        if DISPLAY_PARAMS:
            total_params = 0
            for k, v in {
                "Encoder": self.encoder,
                "Encoder Transformer": self.encoder_trans,
                "Decoder": self.decoder,
            }.items():
                print(f"{k} parameters")
                component_params = 0
                for name, param in v.named_parameters():
                    print(
                        f"Layer: {name}\t Parameters: {param.numel()}\t Trainable: {param.requires_grad}"
                    )
                    component_params += param.numel()
                total_params += component_params
                print(f"Total {k} parameters: {component_params}")

            print(f"Total parameters: {total_params}")

    def training(self, args, epoch):
        # Only need one epoch of hist for printing as is stored in wandb to reduce memory
        self.index_i = 0
        self.hist = np.zeros((args.num_epochs * 2 * len(self.train_loader), 5))

        if self.start_train_goal != 2:
            self.encoder.train()
            self.encoder_trans.train()
            self.decoder.train()  # train mode (dropout and batchnorm is used)
        else:
            if self.args.train_goal == 2:
                self.encoder.train()
                self.encoder_trans.train()
                self.decoder.train()  # train mode (dropout and batchnorm is used)
            elif self.args.train_goal == 1:
                self.encoder.eval()
                self.encoder_trans.fine_tune(self.args.train_goal)
                self.decoder.train()
            else:
                self.encoder.eval()
                self.encoder_trans.fine_tune(self.args.train_goal)
                self.decoder.eval()

        if self.decoder_optimizer is not None:
            self.decoder_optimizer.zero_grad()
        self.encoder_trans_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()

        for id, (imgA, imgB, seg_label, _, _, token, token_len, _) in enumerate(
            self.train_loader
        ):
            # if id == 120:
            #    break
            self.debug_gpu_memory(f"Before batch {id}")
            start_time = time.time()
            accum_steps = 64 // args.train_batchsize

            # Move to GPU, if available
            imgA = imgA.to(DEVICE)
            imgB = imgB.to(DEVICE)
            seg_label = seg_label.to(DEVICE)
            token = token.squeeze(1).to(DEVICE)
            token_len = token_len.to(DEVICE)
            # Forward prop.
            feat1, feat2 = self.encoder(imgA, imgB)
            feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
            if self.args.train_goal != 0:
                scores, caps_sorted, decode_lengths, sort_ind = self.decoder(
                    feat1, feat2, token, token_len
                )
                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]
                scores = pack_padded_sequence(
                    scores, decode_lengths, batch_first=True
                ).data.detach()
                targets = pack_padded_sequence(
                    targets, decode_lengths, batch_first=True
                ).data.detach()
                # Calculate loss
                cap_loss = self.criterion_cap(scores, targets.to(torch.int64))
            det_loss = self.criterion_det(seg_pre, seg_label.to(torch.int64))
            if self.args.train_goal == 0:
                if self.start_train_goal == 2:
                    if epoch < 100:
                        det_loss = det_loss  # / det_loss.detach().item()
                loss = det_loss
            elif self.args.train_goal == 1:
                # if args.train_stage == 's1':
                #     cap_loss = cap_loss / cap_loss.detach().item()
                loss = cap_loss
            else:
                # balance two losses
                if args.train_stage == "s1":
                    scaling = det_loss.detach() / (cap_loss.detach() + 1e-8)
                    scaled_cap_loss = cap_loss * scaling
                    loss = det_loss + scaled_cap_loss
                else:
                    loss = det_loss + cap_loss
            # Back prop.
            loss = loss / accum_steps
            loss.backward()
            # Clip gradients
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(
                    self.decoder.parameters(), args.grad_clip
                )
                torch.nn.utils.clip_grad_value_(
                    self.encoder_trans.parameters(), args.grad_clip
                )
                if self.encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(
                        self.encoder.parameters(), args.grad_clip
                    )

            # Update weights
            if (id + 1) % accum_steps == 0 or (id + 1) == len(self.train_loader):
                if self.decoder_optimizer is not None:
                    self.decoder_optimizer.step()
                self.encoder_trans_optimizer.step()
                if self.encoder_optimizer is not None:
                    # if epoch >10:
                    self.encoder_optimizer.step()

                # Adjust learning rate
                if self.decoder_lr_scheduler is not None:
                    self.decoder_lr_scheduler.step()
                # print(decoder_optimizer.param_groups[0]['lr'])
                self.encoder_trans_lr_scheduler.step()
                if self.encoder_lr_scheduler is not None:
                    # if epoch > 10:
                    self.encoder_lr_scheduler.step()
                    # print(encoder_optimizer.param_groups[0]['lr'])

                if self.decoder_optimizer is not None:
                    self.decoder_optimizer.zero_grad()
                self.encoder_trans_optimizer.zero_grad()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.zero_grad()

            # Keep track of metrics
            self.hist[self.index_i, 0] = time.time() - start_time  # batch_time
            if self.args.train_goal == 0 or self.args.train_goal == 2:
                self.hist[self.index_i, 1] = det_loss.item()  # train_loss
                self.hist[self.index_i, 2] = accuracy(
                    seg_pre.permute(0, 2, 3, 1).reshape(-1, seg_pre.size(1)),
                    seg_label.reshape(-1),
                    1,
                )
            if self.args.train_goal == 1 or self.args.train_goal == 2:
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
            self.debug_gpu_memory(f"After batch {id}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.list_gpu_tensors()

    # One epoch's validation
    def validation(self, epoch):
        word_vocab = self.word_vocab
        self.decoder.eval()  # eval mode (no dropout or batchnorm)
        self.encoder_trans.eval()
        if self.encoder is not None:
            self.encoder.eval()

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
                tqdm(self.val_loader, desc="val_" + "EVALUATING AT BEAM SIZE " + str(1))
            ):
                # Move to GPU, if available
                imgA = imgA.to(DEVICE)
                imgB = imgB.to(DEVICE)
                token_all = token_all.squeeze(0).to(DEVICE)
                # Forward prop.
                if self.encoder is not None:
                    feat1, feat2 = self.encoder(imgA, imgB)
                feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
                if self.args.train_goal != 0 or self.start_train_goal == 2:
                    seq = self.decoder.sample(feat1, feat2, k=1)

                # for segmentation
                if self.args.train_goal != 1 or self.start_train_goal == 2:
                    pred_seg = seg_pre.data.cpu().numpy()
                    seg_label = seg_label.cpu().numpy()
                    pred_seg = np.argmax(pred_seg, axis=1)
                    # Add batch sample into evaluator
                    self.evaluator.add_batch(seg_label, pred_seg)
                # for captioning
                if self.args.train_goal != 0 or self.start_train_goal == 2:
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

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            val_time = time.time() - val_start_time
            # Fast test during the training
            # for segmentation
            if self.args.train_goal != 1 or self.start_train_goal == 2:
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
            if self.args.train_goal != 0 or self.start_train_goal == 2:
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
        if self.start_train_goal != 2:
            if self.args.train_goal == 0:
                Bleu_4 = 0
            if self.args.train_goal == 1:
                mIoU_seg = 0
            if (
                Bleu_4 > self.best_bleu4
                or mIoU_seg > self.MIou
                or Bleu_4 + mIoU_seg > self.Sum_Metric
            ):
                self.best_bleu4 = max(Bleu_4, self.best_bleu4)
                self.MIou = max(mIoU_seg, self.MIou)
                self.Sum_Metric = max(Bleu_4 + mIoU_seg, self.Sum_Metric)
                # save_checkpoint
                state = {
                    "encoder_dict": self.encoder.state_dict(),
                    "encoder_trans_dict": self.encoder_trans.state_dict(),
                    "decoder_dict": self.decoder.state_dict(),
                }
                metric = f"Sum_{round(100000 * self.Sum_Metric)}_MIou_{round(100000 * self.MIou)}_Bleu4_{round(100000 * self.best_bleu4)}"
                model_name = f"{args.data_name}_bts_{args.train_batchsize}_{args.network}_epo_{epoch}_{metric}.pth"
                if epoch > 10:
                    print("Save Model")
                    torch.save(state, os.path.join(args.savepath, model_name))
        # if True:
        elif self.start_train_goal == 2:
            Sum_Metric = mIoU_seg + Bleu_4
            if (
                (self.args.train_goal == 2 and Sum_Metric >= self.Sum_Metric)
                or (self.args.train_goal == 1 and Bleu_4 >= self.best_bleu4)
                or (self.args.train_goal == 0 and mIoU_seg > self.MIou)
            ):  # or Bleu_4+mIoU_seg > self.Sum_Metric
                self.best_bleu4 = (
                    max(Bleu_4, self.best_bleu4)
                    if self.args.train_goal == 1
                    else Bleu_4
                )
                self.MIou = (
                    max(mIoU_seg, self.MIou) if self.args.train_goal == 0 else mIoU_seg
                )
                self.Sum_Metric = (
                    max(Sum_Metric, self.Sum_Metric)
                    if self.args.train_goal == 2
                    else Sum_Metric
                )
                # save_checkpoint
                print("Save Model")
                state = {
                    "encoder_dict": self.encoder.state_dict(),
                    "encoder_trans_dict": self.encoder_trans.state_dict(),
                    "decoder_dict": self.decoder.state_dict(),
                }
                metric = f"Sum_{round(100000*self.Sum_Metric)}_MIou_{round(100000*self.MIou)}_Bleu4_{round(100000*self.best_bleu4)}"
                # metric = f'MIou_{round(10000 * self.MIou)}_Bleu4_{round(10000 * self.best_bleu4)}'
                model_name = f"{self.args.data_name}_bts_{self.args.train_batchsize}_{self.args.network}_epo_{epoch}_{metric}.pth"
                best_model_path = os.path.join(self.args.savepath, model_name)
                if epoch > 10:
                    torch.save(state, best_model_path)
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
        "--allow_unk", type=int, default=1, help="if unknown token is allowed"
    )
    parser.add_argument(
        "--data_name", default="Forest-Change", help="base name shared by data files."
    )

    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id in the training.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="path to checkpoint from stage s1, assert not None when train_stage=s2",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=5,
        help="print training/validation stats every __ batches",
    )
    # Training parameters
    parser.add_argument(
        "--train_goal", type=int, default=2, help="0:det; 1:cap; 2:two tasks"
    )
    parser.add_argument(
        "--train_stage",
        default="s1",
        help="s1: pretrain backbone under two loss;"
        " s2: train two branch respectively",
    )
    parser.add_argument(
        "--fine_tune_encoder",
        type=bool,
        default=True,
        help="whether fine-tune encoder or not",
    )
    parser.add_argument(
        "--train_batchsize", type=int, default=32, help="batch_size for training"
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
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=1e-4,
        help="learning rate for encoder if fine-tuning.",
    )
    parser.add_argument(
        "--decoder_lr", type=float, default=1e-4, help="learning rate for decoder."
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="clip gradients at an absolute value of.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    # Validation
    parser.add_argument(
        "--val_batchsize", type=int, default=1, help="batch_size for validation"
    )
    parser.add_argument("--savepath", default="./models_ckpt/")
    # backbone parameters
    parser.add_argument(
        "--network",
        default="segformer-mit_b1",
        help="define the backbone encoder to extract features",
    )
    parser.add_argument(
        "--encoder_dim",
        type=int,
        default=512,
        help="the dimension of extracted features using backbone ",
    )
    parser.add_argument(
        "--feat_size",
        type=int,
        default=16,
        help="define the output size of encoder to extract features",
    )
    # Model parameters
    parser.add_argument(
        "--n_heads", type=int, default=8, help="Multi-head attention in Transformer."
    )
    parser.add_argument(
        "--n_layers", type=int, default=3, help="Number of layers in AttentionEncoder."
    )
    parser.add_argument("--decoder_n_layers", type=int, default=1)
    parser.add_argument(
        "--feature_dim", type=int, default=512, help="embedding dimension"
    )
    args = parser.parse_args()

    trainer = Trainer(args)
    print_log("\nStarting Epoch: {}".format(trainer.start_epoch), trainer.log)
    print_log("Total Epoches: {}".format(trainer.args.num_epochs), trainer.log)
    print_log(
        "Training Dataset Size: {}".format(trainer.train_dataset_size), trainer.log
    )

    try:
        if args.train_goal == 2:
            # First train both together, then train only change captioning, and finally train only change detection
            for goal in [2, 1, 0]:
                print_log(f"Current train_goal={goal}:\n", trainer.log)
                trainer.args.train_goal = goal
                if goal == 2:
                    trainer.args.train_stage = "s1"
                    trainer.args.checkpoint = None
                    for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
                        trainer.training(trainer.args, epoch)
                        trainer.validation(epoch)
                        if epoch - trainer.best_epoch > trainer.args.patience:
                            print_log(
                                f"Model did not improve after {trainer.args.patience}. Stopping training early.",
                                trainer.log,
                            )
                            trainer.start_epoch = trainer.best_epoch + 1
                            break
                        elif epoch == trainer.args.num_epochs - 1:
                            trainer.start_epoch = trainer.best_epoch + 1
                            trainer.args.num_epochs = (
                                trainer.start_epoch + args.num_epochs
                            )
                else:
                    trainer.args.train_stage = "s2"
                    trainer.args.checkpoint = trainer.best_model_path
                    trainer.build_model()
                    for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
                        trainer.training(trainer.args, epoch)
                        trainer.validation(epoch)
                        if (
                            trainer.args.train_goal == 1
                            and epoch - trainer.best_epoch > trainer.args.patience
                        ):
                            trainer.start_epoch = trainer.best_epoch + 1
                            trainer.args.num_epochs = (
                                trainer.start_epoch + trainer.args.num_epochs
                            )
                            print_log(
                                f"Model did not improve after {trainer.args.patience}. Stopping training early.",
                                trainer.log,
                            )
                            break
                    # trainer.args.num_epochs = trainer.start_epoch + trainer.args.num_epochs
        else:
            for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
                trainer.training(trainer.args, epoch)
                # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
                trainer.validation(epoch)
    except Exception as e:
        print_log("Hit an exception: {}".format(e), trainer.log)
