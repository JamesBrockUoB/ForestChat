"""
test_gpt4v_change_captioning.py

Zero-shot change captioning evaluation using GPT-4V.
Mirrors the structure of test.py — uses the same dataloaders, vocab,
and metadata files, and scores with the same get_eval_score() call.

Usage:

  # Forest-Change (default)
  python test_gpt4v_change_captioning.py \
      --result_path ./predict_results/gpt4v

  # LEVIR-MCI-Trees
  python test_gpt4v_change_captioning.py \
      --data_name LEVIR-MCI-Trees \
      --data_folder ./data/LEVIR-MCI-Trees/images \
      --list_path ./data/LEVIR-MCI-Trees/ \
      --token_folder ./data/LEVIR-MCI-Trees/tokens/ \
      --result_path ./predict_results/gpt4v

  # Evaluate only (skip querying, re-score saved results)
  python test_gpt4v_change_captioning.py --eval_only
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv
from gpt4v_change_captioning import (
    DATASET_NORM,
    DATASET_PROMPTS,
    GPT4VChangeCaptioner,
    _numpy_to_base64,
    build_dataloader,
)
from tqdm import tqdm
from utils_tool.utils import get_eval_score, str2bool

load_dotenv()


def get_output_path(result_path: str, data_name: str, split: str) -> str:
    return os.path.join(result_path, f"gpt4v_{data_name}_{split}_captions.jsonl")


def load_done(output_path: str) -> set:
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["name"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def load_results(output_path: str) -> dict:
    results = {}
    with open(output_path) as f:
        for line in f:
            entry = json.loads(line)
            results[entry["name"]] = entry["caption"]
    return results


def caption_to_token_ids(caption: str, word_vocab: dict) -> list:
    special = {word_vocab.get(t) for t in ("<START>", "<END>", "<NULL>")}
    tokens = caption.strip().rstrip(".").split()
    return [
        word_vocab[t]
        for t in tokens
        if t in word_vocab and word_vocab[t] not in special
    ]


def evaluate(
    results: dict,
    references: list,
    names: list,
    word_vocab: dict,
    data_name: str,
    split: str,
    result_path: str,
):
    """
    Score GPT-4V captions against ground-truth references using get_eval_score(),
    """
    ref_list = []
    hyp_list = []
    missing = []

    for name, img_tokens in zip(names, references):
        if name not in results:
            missing.append(name)
            continue
        hyp_list.append(caption_to_token_ids(results[name], word_vocab))
        ref_list.append(img_tokens)

    if missing:
        print(
            f"Warning: {len(missing)} images have no GPT-4V prediction and will be skipped."
        )

    print(f"\nScoring {len(hyp_list)} captions...")
    score_dict = get_eval_score(ref_list, hyp_list)

    print(
        "\n=== GPT-4V Zero-Shot Change Captioning ===\n"
        f"Dataset : {data_name}  |  Split: {split}\n"
        f"BLEU-1  : {score_dict['Bleu_1']:.5f}\n"
        f"BLEU-2  : {score_dict['Bleu_2']:.5f}\n"
        f"BLEU-3  : {score_dict['Bleu_3']:.5f}\n"
        f"BLEU-4  : {score_dict['Bleu_4']:.5f}\n"
        f"METEOR  : {score_dict['METEOR']:.5f}\n"
        f"ROUGE-L : {score_dict['ROUGE_L']:.5f}\n"
        f"CIDEr   : {score_dict['CIDEr']:.5f}\n"
    )

    score_path = os.path.join(result_path, f"gpt4v_{data_name}_{split}_scores.json")
    with open(score_path, "w") as f:
        json.dump({"dataset": data_name, "split": split, **score_dict}, f, indent=4)
    print(f"Scores saved to: {score_path}")
    return score_dict


def main(args):
    with open(os.path.join(args.list_path, args.vocab_file + ".json")) as f:
        word_vocab = json.load(f)

    with open(os.path.join(args.list_path, args.metadata_file + ".json")) as f:
        max_length = json.load(f)["max_length"]

    os.makedirs(args.result_path, exist_ok=True)
    output_path = get_output_path(args.result_path, args.data_name, args.split)

    loader = build_dataloader(args, max_length)
    mean, std = DATASET_NORM[args.data_name]
    prompt = DATASET_PROMPTS[args.data_name]()

    if not args.eval_only:
        captioner = GPT4VChangeCaptioner(
            api_key=args.api_key or os.environ.get("OPEN_AI_KEY"),
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        done = load_done(output_path)
        total = len(loader.dataset)
        print(
            f"[{args.data_name}] {args.split}: {total} images, "
            f"{len(done)} done, {total - len(done)} remaining."
        )

        with open(output_path, "a") as f_out:
            for imgA, imgB, _, _, _, _, _, name in tqdm(
                loader, desc="GPT-4V captioning"
            ):
                img_name = name[0]

                if img_name in done:
                    continue

                enc_A = _numpy_to_base64(imgA[0].numpy(), mean, std)
                enc_B = _numpy_to_base64(imgB[0].numpy(), mean, std)

                try:
                    caption = captioner.query(enc_A, enc_B, prompt)
                except Exception as e:
                    print(f"\n  [ERROR] {img_name}: {e}")
                    continue

                print(f"\n  {img_name} -> {caption}")

                entry = {
                    "name": img_name,
                    "caption": caption,
                    "split": args.split,
                    "dataset": args.data_name,
                }
                f_out.write(json.dumps(entry) + "\n")
                f_out.flush()
                done.add(img_name)

                if args.delay > 0:
                    time.sleep(args.delay)
    else:
        print(f"--eval_only set. Loading results from {output_path}")

    references = []
    names = []
    special = {word_vocab.get(t) for t in ("<START>", "<END>", "<NULL>")}

    for _, _, _, token_all, _, _, _, name in tqdm(
        build_dataloader(args, max_length), desc="Loading references"
    ):
        img_name = name[0]
        img_token = token_all.squeeze(0).tolist()
        img_tokens = [[w for w in caption if w not in special] for caption in img_token]
        references.append(img_tokens)
        names.append(img_name)

    results = load_results(output_path)
    evaluate(
        results,
        references,
        names,
        word_vocab,
        args.data_name,
        args.split,
        args.result_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot image change captioning with GPT-4V"
    )

    parser.add_argument(
        "--data_name",
        default="Forest-Change",
        choices=["Forest-Change", "LEVIR-MCI-Trees"],
    )
    parser.add_argument("--data_folder", default="./data/Forest-Change-dataset/images")
    parser.add_argument("--list_path", default="./data/Forest-Change/")
    parser.add_argument("--token_folder", default="./data/Forest-Change/tokens/")
    parser.add_argument("--vocab_file", default="vocab")
    parser.add_argument("--metadata_file", default="metadata")
    parser.add_argument("--allow_unk", type=str2bool, default=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)

    # Output
    parser.add_argument("--result_path", default="./predict_results/gpt4v")

    # API
    parser.add_argument(
        "--api_key",
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)",
    )
    parser.add_argument("--model", default="gpt-4-vision-preview")
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls to avoid rate limits",
    )

    # Control
    parser.add_argument(
        "--eval_only",
        type=str2bool,
        default=False,
        help="Skip querying and only score already-saved results",
    )

    args = parser.parse_args()
    main(args)
