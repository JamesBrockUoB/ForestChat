"""
gpt4o_change_captioning.py

Zero-shot image change captioning using GPT-4o.
Supports Forest-Change and LEVIR-MCI-Trees datasets.

Also enables trained model caption refinement using GPT-4o
"""

import base64
import logging
import os
import tempfile
from typing import Optional

import cv2
import numpy as np
import requests
from data.ForestChange import NORMALISATION_MEAN as FOREST_MEAN
from data.ForestChange import NORMALISATION_STD as FOREST_STD
from data.ForestChange import ForestChangeDataset
from data.LEVIRMCITrees import NORMALISATION_MEAN as LEVIR_MEAN
from data.LEVIRMCITrees import NORMALISATION_STD as LEVIR_STD
from data.LEVIRMCITrees import LEVIRMCITreesDataset
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from torch.utils import data

logger = logging.getLogger(__name__)

DATASET_NORM = {
    "Forest-Change": (FOREST_MEAN, FOREST_STD),
    "LEVIR-MCI-Trees": (LEVIR_MEAN, LEVIR_STD),
}


def _numpy_to_base64(img_chw: np.ndarray, mean, std) -> str:
    """
    Reverse the normalisation applied in the dataloader, convert CHW -> HWC,
    write to a temp PNG and return as base64.
    """
    img = img_chw.copy().astype(np.float32)  # (C, H, W)
    for c in range(img.shape[0]):
        img[c] = img[c] * std[c] + mean[c]
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)  # (H, W, C)  RGB
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, img_bgr)
    with open(tmp_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    os.remove(tmp_path)
    return encoded


SYSTEM_MESSAGE = (
    "You are an expert analyst of satellite and aerial imagery specialising in "
    "land cover and urban change detection. You will be shown two images of the "
    "same location taken at different times. Always describe only what has "
    "genuinely changed between Image A (before) and Image B (after). "
    "Be factual and concise."
)


def get_forest_change_prompt() -> str:
    return (
        "You are given two aerial forest images of the same location taken at different times.\n"
        "Image A is 'before' and Image B is 'after'.\n\n"
        "Generate ONE caption describing any forest cover change between the images.\n\n"
        "Study these example captions carefully and match their style exactly:\n"
        "- 'minor forest loss is visible found scattered in numerous small patches "
        "which are highly varied in size largely concentrated in the bottom-left and center sections'\n"
        "- 'primarily occurring in the top-center and middle-left regions some modest "
        "forest loss is detected found scattered in numerous small patches which are highly varied in size'\n"
        "- 'mainly located across the top-center and middle-left areas some modest forest loss is detected'\n"
        "- 'low levels of forest degradation are observed occurring in many small patches "
        "which are displaying large variations in size primarily occurring in the top-center and middle-left regions'\n"
        "- 'limited deforestation observed largely concentrated in the bottom-left and center "
        "sections occurring in many small patches which are highly varied in size'\n"
        "- 'the scene is the same as before' (use ONLY if there is genuinely no change)\n\n"
        "Style rules:\n"
        "1. The sentence has two main components that can appear in EITHER ORDER:\n"
        "   - CHANGE component: severity word ('minor', 'limited', 'low levels of', "
        "'some modest', 'slight', 'heavy', 'significant') + change noun ('forest loss', "
        "'forest degradation', 'deforestation') + verb phrase ('is visible', 'is detected', "
        "'is apparent', 'are observed', 'is noted', 'observed').\n"
        "   - LOCATION component: introduced by 'largely concentrated in', 'primarily occurring in', "
        "'mainly located across', or 'found scattered in', followed by compound region terms "
        "such as 'top-center', 'bottom-left', 'middle-left', 'top-right', 'center'.\n"
        "2. Optionally add a PATCH component: 'occurring in many small patches which are "
        "highly varied/moderately varied in size' or 'displaying large variations in size'.\n"
        "3. Do NOT use cardinal directions (north, south, east, west).\n"
        "4. Output a single sentence with no full stop."
    )


def get_levir_mci_prompt() -> str:
    return (
        "You are given two aerial images of the same urban or suburban location taken at different times.\n"
        "Image A is 'before' and Image B is 'after'.\n\n"
        "Generate ONE caption describing what has changed between the two images.\n\n"
        "Study these example captions carefully and match their style exactly:\n"
        "- 'some buildings appear at the bottom of the scene .'\n"
        "- 'a row of houses with streets connected to the road is built .'\n"
        "- 'a new road appears in the center and some trees are removed .'\n"
        "- 'some houses are built on the bareland and some houses appear to replace the trees .'\n"
        "- 'more buildings show up besides the roads and some trees are removed .'\n"
        "- 'two houses appear on both sides of the original house .'\n"
        "- 'the scene is the same as before .' (use ONLY if there is genuinely no change)\n\n"
        "Style rules:\n"
        "1. Be concise — typically one short sentence, occasionally two clauses joined with 'and'.\n"
        "2. Use present-tense or passive constructions: 'appears', 'are built', "
        "'is constructed', 'are removed', 'show up', 'disappear', 'has appeared'.\n"
        "3. Refer to specific object types: 'houses', 'buildings', 'villas', "
        "'road', 'lane', 'bareland', 'trees', 'plants'.\n"
        "4. Quantify where possible: 'some', 'many', 'a row of', 'two', 'several'.\n"
        "5. Locate changes simply: 'at the bottom', 'in the center', 'on the left', "
        "'in the top right corner', 'beside the road', 'on the bareland'.\n"
        "6. End the sentence with a space then a full stop: ' .'\n"
        "7. Do NOT use cardinal directions (north, south, east, west).\n"
        "8. Do NOT describe unchanged background context."
    )


def get_general_prompt() -> str:
    return (
        "You are given two aerial images of the same location taken at different times.\n"
        "Image A is 'before' and Image B is 'after'.\n\n"
        "Describe what has changed between the two images in one sentence. "
        "If there is no change, say 'the scene is the same as before'.\n\n"
        "Rules:\n"
        "1. Output a single sentence only.\n"
        "2. Be specific about what has changed and where.\n"
        "3. Do not describe things that have not changed."
        "4. Do not refer to 'Image A' or 'Image B'."
    )


DATASET_PROMPTS = {
    "General": get_general_prompt,
    "Forest-Change": get_forest_change_prompt,
    "LEVIR-MCI-Trees": get_levir_mci_prompt,
}


class GPT4oChangeCaptioner:
    """Queries GPT-4o with a before/after image pair for zero-shot change captioning."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 300,
        temperature: float = 0.2,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No OpenAI API key provided. Set OPENAI_API_KEY env var "
                "or pass --api_key."
            )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def query(self, base64_A: str, base64_B: str, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image A (before):"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_A}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": "Image B (after):"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_B}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            raise IOError(f"Network error: {type(e).__name__}: {e}") from e

        if not response.ok:
            raise IOError(
                f"GPT-4o request failed [{response.status_code}]: {response.json()}"
            )

        remaining = response.headers.get("x-ratelimit-remaining-requests", "?")
        reset = response.headers.get("x-ratelimit-reset-requests", "?")
        print(f"  API: {remaining} requests remaining, reset in {reset}")

        return response.json()["choices"][0]["message"]["content"].strip()


def build_dataloader(args, max_length: int):
    """
    Build the appropriate DataLoader for the dataset, exactly as in test.py.
    Batch size is fixed to 1 since we encode each image individually for the API.
    """
    common_kwargs = dict(
        data_folder=args.data_folder,
        list_path=args.list_path,
        split=args.split,
        token_folder=args.token_folder,
        vocab_file=args.vocab_file,
        max_length=max_length,
        allow_unk=args.allow_unk,
        num_classes=args.num_classes,
    )

    if "Forest-Change" in args.data_name:
        dataset = ForestChangeDataset(**common_kwargs)
    elif "LEVIR-MCI-Trees" in args.data_name:
        dataset = LEVIRMCITreesDataset(**common_kwargs)
    else:
        raise ValueError(
            f"Unknown dataset '{args.data_name}'. "
            f"Choose from: {list(DATASET_PROMPTS.keys())}"
        )

    loader = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return loader


# -----------------------------------------------------------------------------
# Refinement: spatially enrich a trained-model caption using GPT-4o
# -----------------------------------------------------------------------------


def get_forest_refine_prompt(predicted_caption: str) -> str:
    return (
        "A trained model produced the following change caption for these two aerial forest images:\n\n"
        f'  PREDICTED: "{predicted_caption}"\n\n'
        "The predicted caption captures the change type and severity well but may lack "
        "precise spatial grounding, forest-specific detail, or reasoning about the nature of the change.\n\n"
        "Your task: rewrite the caption so it KEEPS the severity word and change noun from the "
        "prediction (e.g. 'minor', 'forest loss') and enriches it using what you can directly observe "
        "in the images. You may draw on any of the following where visible and relevant:\n"
        "  - Spatial location of the change (region terms such as 'top-center', 'bottom-left', etc.) or relation to geographic features "
        " (e.g. roads, rivers, existing clearings, etc.)\n"
        "  - Patch characteristics (size variation, number, distribution)\n"
        "  - Forest cover context (canopy density, tree cover fragmentation, understorey exposure)\n"
        "  - Likely change process (e.g. selective logging suggested by irregular patch edges, "
        "clearcut suggested by sharp rectangular boundaries, windthrow suggested by fallen crown patterns)\n\n"
        "Study these example captions carefully and match their style exactly:\n"
        "- 'minor forest loss is visible found scattered in numerous small patches "
        "which are highly varied in size largely concentrated in the bottom-left and center sections'\n"
        "- 'primarily occurring in the top-center and middle-left regions some modest "
        "forest loss is detected found scattered in numerous small patches which are highly varied in size'\n"
        "- 'mainly located across the top-center and middle-left areas some modest forest loss is detected'\n"
        "- 'low levels of forest degradation are observed occurring in many small patches "
        "which are displaying large variations in size primarily occurring in the top-center and middle-left regions'\n"
        "- 'limited deforestation observed largely concentrated in the bottom-left and center "
        "sections occurring in many small patches which are highly varied in size'\n"
        "- 'the scene is the same as before' (use ONLY if there is genuinely no change)\n\n"
        "Style rules:\n"
        "1. SEVERITY + CHANGE NOUN: preserve from the prediction unless clearly wrong.\n"
        "2. LOCATION: use compound region terms — 'top-center', 'bottom-left', 'middle-right', etc.\n"
        "3. Optionally add a PATCH component: 'occurring in many small patches which are "
        "highly varied/moderately varied in size' or 'displaying large variations in size'.\n"
        "4. Optionally add a brief forest or process descriptor where clearly supported by the imagery "
        "(e.g. 'with exposed understorey', 'consistent with selective logging', 'along forest edges').\n"
        "5. Do NOT use cardinal directions (north, south, east, west).\n"
        "6. Do NOT speculate beyond what is visible.\n"
        "7. Output a single sentence with no full stop."
    )


def get_levir_refine_prompt(predicted_caption: str) -> str:
    return (
        "A trained model produced the following change caption for these two aerial urban images:\n\n"
        f'  PREDICTED: "{predicted_caption}"\n\n'
        "The prediction captures the change objects well but may miss spatial detail, secondary changes, "
        "or contextual reasoning about the development visible in the images.\n\n"
        "Your task: rewrite the caption to ADD precise location references and any additional "
        "visible changes while KEEPING the object types and change verbs from the prediction. "
        "You may also draw on the following where visible and relevant:\n"
        "  - Infrastructure context (e.g. road access enabling construction, proximity to existing buildings)\n"
        "  - Land cover transitions (e.g. bareland cleared prior to construction, vegetation removed)\n"
        "  - Development pattern (e.g. infill development, expansion along a road, estate-style layout)\n\n"
        "Study these example captions carefully and match their style exactly:\n"
        "- 'some buildings appear at the bottom of the scene .'\n"
        "- 'a row of houses with streets connected to the road is built .'\n"
        "- 'a new road appears in the center and some trees are removed .'\n"
        "- 'some houses are built on the bareland and some houses appear to replace the trees .'\n"
        "- 'more buildings show up besides the roads and some trees are removed .'\n"
        "- 'two houses appear on both sides of the original house .'\n"
        "- 'the scene is the same as before .' (use ONLY if there is genuinely no change)\n\n"
        "Style rules:\n"
        "1. Keep object types from the prediction unless clearly wrong.\n"
        "2. Add location: 'at the bottom', 'in the center', 'on the left', 'beside the road'.\n"
        "3. Use present-tense or passive: 'appears', 'are built', 'is constructed', 'are removed'.\n"
        "4. Quantify: 'some', 'a row of', 'two', 'several'.\n"
        "5. Optionally add a brief context clause where clearly supported by the imagery "
        "(e.g. 'on previously cleared land', 'alongside the existing road', 'replacing sparse vegetation').\n"
        "6. End the sentence with a space then a full stop: ' .'\n"
        "7. Do NOT use cardinal directions (north, south, east, west).\n"
        "8. Do NOT speculate beyond what is visible.\n"
        "9. Output a single sentence only."
    )


DATASET_REFINE_PROMPTS = {
    "Forest-Change": get_forest_refine_prompt,
    "LEVIR-MCI-Trees": get_levir_refine_prompt,
}


class GPT4oCaptionRefiner:
    """Queries GPT-4o to spatially enrich a trained-model change caption."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 300,
        temperature: float = 0.2,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No OpenAI API key provided. Set OPENAI_API_KEY env var "
                "or pass --api_key."
            )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def refine(
        self,
        base64_A: str,
        base64_B: str,
        predicted_caption: str,
        dataset: str = "Forest-Change",
    ) -> str:
        if dataset not in DATASET_REFINE_PROMPTS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Choose from: {list(DATASET_REFINE_PROMPTS)}"
            )

        prompt = DATASET_REFINE_PROMPTS[dataset](predicted_caption)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image A (before):"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_A}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": "Image B (after):"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_B}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            raise IOError(f"Network error: {type(e).__name__}: {e}") from e

        if not response.ok:
            raise IOError(
                f"GPT-4o request failed [{response.status_code}]: {response.json()}"
            )

        remaining = response.headers.get("x-ratelimit-remaining-requests", "?")
        reset = response.headers.get("x-ratelimit-reset-requests", "?")
        print(f"  API: {remaining} requests remaining, reset in {reset}")

        return response.json()["choices"][0]["message"]["content"].strip()
