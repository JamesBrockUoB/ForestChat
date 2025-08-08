import json
import os
import random
import re

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_file_browser import st_file_browser

LOSS_ADJECTIVES = {
    (0, 1): [
        "negligible forest loss is detected",
        "almost no forest change is observed",
        "very minimal forest degradation",
    ],
    (1, 5): [
        "minor forest loss is visible",
        "slight forest degradation is noted",
        "small amounts of forest loss are apparent",
        "limited deforestation observed",
    ],
    (5, 10): [
        "low levels of forest degradation are observed",
        "some modest forest loss is detected",
        "slight forest loss is apparent",
    ],
    (10, 15): [
        "moderate forest loss is present",
        "noticeable forest degradation occurs",
        "intermediate deforestation is evident",
    ],
    (15, 25): [
        "substantial deforestation has occurred",
        "considerable forest loss is visible",
        "significant forest loss is observed",
    ],
    (25, 40): [
        "extensive forest loss is detected in the observed area",
        "large-scale forest degradation is apparent",
        "widespread clearings are present",
    ],
    (40, 65): [
        "severe deforestation affects a large portion of the scene",
        "major forest loss dominates the area",
        "heavy deforestation is visible throughout much of the region",
    ],
    (65, 85): [
        "critical forest loss dominates the region",
        "an overwhelming level of deforestation is evident",
        "very high levels of deforestation detected",
    ],
    (85, 100): [
        "near-total deforestation is observed throughout the area",
        "almost complete loss of forest detected",
        "forest cover is nearly entirely removed",
    ],
}

TEMPLATES = [
    "{adj} {distribution} {patchiness}",
    "{adj} {patchiness} {distribution}",
    "{distribution} {adj} {patchiness}",
    "{adj} {distribution}",
    "{distribution} {adj}",
]
TEMPLATE_WEIGHTS = [0.35, 0.35, 0.3, 0.15, 0.15]


class AutoCaption:
    def __init__(self, example):
        self.example = example
        self.img = self.load_image()

    def load_image(self):
        img = cv2.imread(
            self.example.path_mask, cv2.IMREAD_GRAYSCALE
        )  # Read in grayscale
        if img is None:
            print(f"Could not read image {self.example.path_mask}")
            return None

        return img

    def describe_forest_loss_distribution(self, img):
        h, w = img.shape
        patch_counts = np.zeros((3, 3))
        patch_h, patch_w = h // 3, w // 3
        for i in range(3):
            for j in range(3):
                patch = img[
                    i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w
                ]
                patch_counts[i, j] = np.sum(patch == 255)

        total_change = np.sum(patch_counts)
        if total_change == 0:
            return "no forest change is detected"

        patch_percentage = patch_counts / total_change
        flat = patch_percentage.flatten()

        sorted_idx = flat.argsort()[::-1]
        positions = [
            "top-left",
            "top-center",
            "top-right",
            "middle-left",
            "center",
            "middle-right",
            "bottom-left",
            "bottom-center",
            "bottom-right",
        ]

        if flat[sorted_idx[0]] > 0.5:
            return random.choice(
                [
                    f"concentrated in the {positions[sorted_idx[0]]} region",
                    f"mainly located in the {positions[sorted_idx[0]]} area",
                    f"primarily found in the {positions[sorted_idx[0]]} section",
                ]
            )
        elif np.count_nonzero(flat > 0.15) >= 4:
            return random.choice(
                [
                    "scattered across multiple regions",
                    "distributed throughout various parts of the area",
                    "spread across several different regions",
                ]
            )
        else:
            return random.choice(
                [
                    f"primarily occurring in the {positions[sorted_idx[0]]} and {positions[sorted_idx[1]]} regions",
                    f"mainly located across the {positions[sorted_idx[0]]} and {positions[sorted_idx[1]]} areas",
                    f"largely concentrated in the {positions[sorted_idx[0]]} and {positions[sorted_idx[1]]} sections",
                ]
            )

    def describe_patchiness_of_loss(self, img):
        total_pixels = img.shape[0] * img.shape[1]
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

        if num_labels <= 1:
            return "with no visible forest change"

        patch_areas = stats[1:, cv2.CC_STAT_AREA]
        patch_percents = patch_areas / total_pixels * 100
        avg = np.mean(patch_percents)
        std = np.std(patch_percents)
        std_rate = std / avg if avg > 0 else 0

        if std_rate < 0.25:
            var = random.choice(["uniform in size", "relatively consistent in size"])
        elif std_rate < 0.6:
            var = random.choice(
                ["moderately varied in size", "showing some variation in size"]
            )
        else:
            var = random.choice(
                ["highly varied in size", "displaying large variations in size"]
            )

        if num_labels <= 3:
            return random.choice(
                [
                    f"in a few patches that are {var}",
                    f"within a small number of patches which are {var}",
                ]
            )

        if np.any(patch_percents >= 10):
            return random.choice(
                [
                    f"in several patches, including one or more extensive regions, which are {var}",
                    f"across multiple patches with some large areas, which are {var}",
                ]
            )
        if np.any(patch_percents >= 5):
            return random.choice(
                [
                    f"in multiple patches, including some notable regions, which are {var}",
                    f"distributed among several patches with some notable sizes, which are {var}",
                ]
            )

        return random.choice(
            [
                f"occurring in many small patches, which are {var}",
                f"found scattered in numerous small patches, which are {var}",
            ]
        )

    def select_adjective(self, percentage):
        for (low, high), phrases in LOSS_ADJECTIVES.items():
            if low <= percentage < high:
                return random.choice(phrases)
        return "forest loss is detected"

    def calculate_deforestation_percentage(self):
        """Calculate percentage from mask with error handling"""
        try:

            total_pixels = self.img.shape[0] * self.img.shape[1]
            deforestation_pixels = np.sum(self.img != 0)
            return round((deforestation_pixels / total_pixels) * 100.0, 2)
        except Exception:
            return None

    def remove_duplicate_adjacent_words(self, text):
        # replace repeated adjacent word sequences (case-insensitive)
        # e.g. "the the" -> "the", "and and" -> "and"
        text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
        # collapse multiple spaces and fix spacing before punctuation
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,\.])", r"\1", text)
        return text

    def generate_auto_captions(self):
        try:
            percentage = self.calculate_deforestation_percentage()
            if percentage is None:
                return None

            captions = []
            for _ in range(4):
                adj = self.select_adjective(percentage)
                distribution = self.describe_forest_loss_distribution(self.img)
                patchiness = self.describe_patchiness_of_loss(self.img)

                adj = "" if adj is None else str(adj).strip()
                distribution = "" if distribution is None else str(distribution).strip()
                patchiness = "" if patchiness is None else str(patchiness).strip()

                template = random.choices(TEMPLATES, weights=TEMPLATE_WEIGHTS, k=1)[0]
                caption = template.format(
                    adj=adj, distribution=distribution, patchiness=patchiness
                )
                caption = self.remove_duplicate_adjacent_words(caption)
                captions.append(caption)

            return captions

        except Exception as e:
            print(f"Error generating captions: {str(e)}")
            return None


class Example:
    def __init__(self, split, filename, path_A, path_B, path_mask=None):
        self.split = split or "unsplit"
        self.filename = filename
        self.path_A = path_A
        self.path_B = path_B
        self.path_mask = path_mask


class DatasetLoader:
    """
    Assumes nested data folder with structure of one of the two below:

    dataset_root/
    └── images/
        ├── train/
        │   ├── A/
        │   ├── B/
        │   └── label/ ← optional
        ├── val/
        │   ├── A/
        │   ├── B/
        │   └── label/ ← optional
        └── test/
            ├── A/
            ├── B/
            └── label/ ← optional

    OR

    dataset_root/
    └── images/
        ├── A/
        ├── B/
        └── label/ ← optional
    """

    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.examples = []
        self._load_examples()

    def _load_examples(self):
        if not os.path.isdir(self.base_folder):
            return

        split_folders = [
            name
            for name in ["train", "val", "test"]
            if os.path.isdir(os.path.join(self.base_folder, "images", name))
        ]

        if split_folders:
            for split in split_folders:
                self.examples.extend(self._load_from_split(split))
        else:
            self.examples.extend(self._load_from_split(None))

    def _load_from_split(self, split):
        examples = []
        base = (
            os.path.join(self.base_folder, "images", split)
            if split
            else self.base_folder
        )
        folder_A = os.path.join(base, "A")
        folder_B = os.path.join(base, "B")
        folder_mask = os.path.join(base, "label")
        has_mask = os.path.isdir(folder_mask)

        if not (os.path.isdir(folder_A) and os.path.isdir(folder_B)):
            return []

        for filename in os.listdir(folder_A):
            if filename.startswith("."):
                continue

            path_B = os.path.join(folder_B, filename)
            if not os.path.exists(path_B):
                continue

            examples.append(
                Example(
                    split=split,
                    filename=filename,
                    path_A=os.path.join(folder_A, filename),
                    path_B=path_B,
                    path_mask=os.path.join(folder_mask, filename) if has_mask else None,
                )
            )

        return examples


class CaptionManager:
    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.caption_file = os.path.join(base_folder, "ForestChatcaptions.json")
        self.captions = self._load_captions()

    def _load_captions(self):
        if os.path.exists(self.caption_file):
            with open(self.caption_file, "r") as f:
                return json.load(f)
        return {"images": []}

    def safe_tokenize(self, text, max_length=250, max_tokens=75):
        tokens = []
        for word in text[:max_length].strip().lower().split():
            if word.replace(".", "", 1).isdigit():
                tokens.append(word)
            else:
                clean_word = "".join(
                    [c for c in word if c.isalpha() or c in ["'", "-"]]
                )
                if clean_word:
                    tokens.append(clean_word)

        return tokens[:max_tokens]

    def save_caption(self, example, caption_text):
        entry = self._get_caption(example.filename)

        if entry is None:
            entry = {
                "filename": example.filename,
                "filepath": example.split,
                "split": example.split,
                "sentences": [],
            }
            self.captions["images"].append(entry)

        existing_raws = [s["raw"] for s in entry["sentences"]]
        if caption_text in existing_raws:
            return

        entry["sentences"].append(
            {
                "raw": caption_text,
                "tokens": self.safe_tokenize(caption_text),
            }
        )

        with open(self.caption_file, "w") as f:
            json.dump(self.captions, f, indent=2)

    def _get_caption(self, example_key):
        return next(
            (img for img in self.captions["images"] if img["filename"] == example_key),
            None,
        )

    def get_labelled_examples(self):
        return {img["filename"] for img in self.captions["images"]}


class CaptioningApp:
    def __init__(self):
        st.set_page_config(
            layout="wide",
            page_title="ForestChat-change-captioning",
            page_icon="🌳",
        )

        st.header("🌳 :blue[ForestChat] 🌲 Change Captioning Tool ", divider="rainbow")
        self._initialise_session_state()
        self.base_folder = self._select_base_folder()

        if self.base_folder:
            self.caption_mgr = CaptionManager(self.base_folder)
            self._load_dataset()
            self.run()

    def _initialise_session_state(self):
        if "show_browser" not in st.session_state:
            st.session_state.show_browser = False

        if "dataset_folder" not in st.session_state:
            st.session_state.dataset_folder = None

        if "temp_selected_path" not in st.session_state:
            st.session_state.temp_selected_path = None

        if "current_index" not in st.session_state:
            st.session_state.current_index = 0

        if "skip_labelled" not in st.session_state:
            st.session_state.skip_labelled = False

        if "skipped_count" not in st.session_state:
            st.session_state.skipped_count = 0

        if "total_examples_count" not in st.session_state:
            st.session_state.total_examples_count = 0

        if "use_mask_captions" not in st.session_state:
            st.session_state.use_mask_captions = False

        if "examples" not in st.session_state:
            st.session_state.examples = []

    def _select_base_folder(self):
        if not st.session_state.show_browser:
            if st.button("Choose Folder"):
                st.session_state.show_browser = True
                st.rerun()
            return st.session_state.get("dataset_folder", None)

        st.checkbox(
            "Skip already labeled files",
            value=st.session_state.skip_labelled,
            key="skip_labelled",
            help="Will skip files that already have captions",
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(
                f"Selected: `{st.session_state.temp_selected_path}`"
                if st.session_state.temp_selected_path
                else "No folder selected yet"
            )

        with col2:
            if st.button(
                "Confirm Folder", disabled=not st.session_state.temp_selected_path
            ):
                self._confirm_folder_selection()

        st.info("Browse and select a folder below:")
        self._handle_file_browser()
        return st.session_state.dataset_folder

    def _confirm_folder_selection(self):
        selected_path = st.session_state.temp_selected_path
        images_path = os.path.join(selected_path, "images")

        if not os.path.isdir(selected_path):
            st.error("Selected path is not a directory.")
        elif not os.path.isdir(images_path):
            st.error("Folder must contain an 'images/' subdirectory.")
        else:
            st.session_state.dataset_folder = selected_path
            st.session_state.show_browser = False
            st.session_state.temp_selected_path = None
            st.rerun()

    def _handle_file_browser(self):
        selection = st_file_browser(
            key="dataset_folder_picker",
            path="./",
            glob_patterns="**/*",
            show_choose_file=True,
            show_download_file=False,
            show_delete_file=False,
            show_upload_file=False,
            show_new_folder=False,
            show_preview=False,
        )

        if selection and "target" in selection:
            st.session_state.temp_selected_path = selection["target"]["path"]
            st.rerun()

    def _load_dataset(self):
        if len(st.session_state.examples) == 0:
            loader = DatasetLoader(self.base_folder)
            st.session_state.examples = loader.examples
            st.session_state.total_examples_count = len(st.session_state.examples)

        if st.session_state.skip_labelled:
            labelled_names = self.caption_mgr.get_labelled_examples()
            filtered = [
                ex
                for ex in st.session_state.examples
                if ex.filename not in labelled_names
            ]
            st.session_state.skipped_count = len(st.session_state.examples) - len(
                filtered
            )
            st.session_state.examples = filtered

            if not st.session_state.examples:
                st.warning(
                    "No images to label!"
                    if not st.session_state.skip_labelled
                    else "All images are already labeled!"
                )
                st.stop()

    def run(self):
        if len(st.session_state.examples) == 0:
            st.error("No valid examples found.")
            return

        example = st.session_state.examples[st.session_state.current_index]

        self._show_progress()

        with st.container(height=525):
            self._display_example(example)
        self._input_caption(example)

    def _show_progress(self):
        current = st.session_state.current_index + st.session_state.skipped_count
        st.sidebar.progress(current / st.session_state.total_examples_count)
        st.sidebar.write(
            f"{current}/{st.session_state.total_examples_count} (Skipped: {st.session_state.skipped_count})"
        )

    def _display_example(self, example):
        st.markdown(f"**Captioning:** `{example.filename}` in `{example.split}`")

        cols = st.columns(3 if example.path_mask else 2)

        cols[0].image(
            Image.open(example.path_A), caption="Before", use_container_width=True
        )
        cols[1].image(
            Image.open(example.path_B), caption="After", use_container_width=True
        )
        if example.path_mask:
            cols[2].image(
                Image.open(example.path_mask),
                caption="Change Mask",
                use_container_width=True,
            )

    def _input_caption(self, example):
        with st.form("caption_form", clear_on_submit=True):
            if example.path_mask:
                st.session_state.use_mask_captions = st.checkbox(
                    "Generate automatic supplementary mask-based captions",
                    value=st.session_state.use_mask_captions,
                    key=f"mask_cb_{st.session_state.current_index}",
                )

            caption = st.text_input(
                "Enter caption:", key=f"caption_{st.session_state.current_index}"
            )

            if st.form_submit_button("Submit Caption"):
                if not caption.strip():
                    st.warning("Please enter a caption.")
                    return

                self.caption_mgr.save_caption(example, caption.strip())

                if example.path_mask and st.session_state.use_mask_captions:
                    auto_captions = AutoCaption(example).generate_auto_captions()
                    if auto_captions:
                        for cap in auto_captions:
                            self.caption_mgr.save_caption(example, cap)

                st.session_state.current_index += 1
                if st.session_state.current_index >= len(st.session_state.examples):
                    st.success("Labelling complete!")
                    st.session_state.show_browser = False
                    st.stop()
                st.rerun()


if __name__ == "__main__":
    CaptioningApp()
