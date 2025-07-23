import json
import os
from string import punctuation

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_file_browser import st_file_browser


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

    def create_captions(self):
        deforestation_rate = self.check_percentage_of_image_contains_deforestation()

        caption_percentage = self.create_deforestation_percentage_caption(
            deforestation_rate
        )
        caption_adjective = self.loss_intensity_adjective_caption(deforestation_rate)
        caption_distribution_of_loss = self.describe_forest_loss_distribution()
        caption_deforestation_patchiness = self.describe_patchiness_of_loss()
        deforestation_adjective_distribution = self.combine_two_captions(
            caption_adjective, caption_distribution_of_loss
        )
        deforestation_adjective_patchiness = self.combine_two_captions(
            caption_adjective, caption_deforestation_patchiness
        )

        return [
            caption_percentage,
            caption_adjective,
            deforestation_adjective_distribution,
            deforestation_adjective_patchiness,
        ]

    def check_percentage_of_image_contains_deforestation(self):
        total_pixels = self.img.shape[0] * self.img.shape[1]
        deforestation_pixels = np.sum(self.img == 255)
        percentage = (deforestation_pixels / total_pixels) * 100.0
        return round(percentage, 2)

    def create_deforestation_percentage_caption(self, deforestation_rate):
        return f"{deforestation_rate} percent of the observed area has been affected by deforestation"

    def loss_intensity_adjective_caption(self, percentage):
        if percentage < 1:
            return "negligible forest loss is detected"
        elif percentage < 5:
            return "minor forest loss is visible"
        elif percentage < 10:
            return "low levels of forest degradation are observed"
        elif percentage < 20:
            return "moderate forest loss is present"
        elif percentage < 35:
            return "substantial deforestation has occurred"
        elif percentage < 50:
            return "extensive forest loss is detected in the observed area"
        elif percentage < 75:
            return "severe deforestation affects a large portion of the scene"
        elif percentage < 90:
            return "critical forest loss dominates the region"
        else:
            return "near-total deforestation is observed throughout the observed area"

    def describe_forest_loss_distribution(self):
        h, w = self.img.shape
        patch_counts = np.zeros((3, 3))
        patch_h, patch_w = h // 3, w // 3

        for i in range(3):
            for j in range(3):
                patch = self.img[
                    i * patch_h : (i + 1) * patch_h,
                    j * patch_w : (j + 1) * patch_w,
                ]
                patch_counts[i, j] = np.sum(patch == 255)

        total_change = np.sum(patch_counts)
        if total_change == 0:
            return "no forest change detected"

        patch_percentage = patch_counts / total_change
        flat = patch_percentage.flatten()
        sorted_idx = flat.argsort()[::-1]

        if flat[0] > 0.5:
            region = self.grid_position_name(sorted_idx[0])
            return f"concentrated in the {region}"
        elif np.count_nonzero(flat > 0.15) >= 4:
            return "scattered across the image in multiple regions"

        else:
            dominant_regions = [self.grid_position_name(idx) for idx in sorted_idx[:2]]
            return f"primarily occurring in the {dominant_regions[0]} and {dominant_regions[1]} regions"

    def grid_position_name(self, idx):
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
        return positions[idx]

    def describe_patchiness_of_loss(self):
        total_pixels = self.img.shape[0] * self.img.shape[1]
        _, binary = cv2.threshold(self.img, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        patch_areas = stats[1:, cv2.CC_STAT_AREA]

        if len(patch_areas) == 0:
            return "with no visible forest change"

        patch_areas = np.array(patch_areas)
        patch_percents = patch_areas / total_pixels * 100

        very_small = np.sum((patch_percents >= 0.1) & (patch_percents < 0.5))
        small = np.sum((patch_percents >= 0.5) & (patch_percents < 2))
        moderate = np.sum((patch_percents >= 2) & (patch_percents < 5))
        notable = np.sum((patch_percents >= 5) & (patch_percents < 10))
        extensive = np.sum(patch_percents >= 10)

        avg_area = np.mean(patch_percents)
        std_area = np.std(patch_percents)
        std_rate = std_area / avg_area if avg_area > 0 else 0

        if std_rate < 0.25:
            variability = "uniform in size"
        elif std_rate < 0.6:
            variability = "moderately varied in size"
        else:
            variability = "highly varied in size"

        total_patches = len(patch_percents)

        description = "occurring in "

        if total_patches <= 3:
            return f"occurring in a few patches that are {variability}"

        if extensive > 0:
            description += "several patches, including one or more extensive regions"
        elif notable > 0:
            description += "multiple patches, including some notable regions"
        elif moderate > 0:
            description += "many moderate-sized patches"
        elif small > 0 or very_small > 0:
            description += "numerous small patches"
        else:
            description += "many tiny patches"

        description += f", which are {variability}"

        return description

    def combine_two_captions(self, caption_one, caption_two):
        return f"{caption_one}, {caption_two}"


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

    def safe_tokenize(self, text, max_length=150, max_tokens=75):
        tokens = []
        for word in text.strip().lower().split():
            if word.replace(".", "", 1).isdigit():
                tokens.append(word)
            else:
                clean_word = "".join([c for c in word if c.isalpha() or c == "'"])
                if clean_word:
                    tokens.append(clean_word)

        return tokens

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

        if "use_mask_captions" not in st.session_state:
            st.session_state.use_mask_captions = False

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
        loader = DatasetLoader(self.base_folder)
        self.examples = loader.examples

        if st.session_state.skip_labelled:
            labelled = self.caption_mgr.get_labelled_examples()
            filtered = [ex for ex in self.examples if ex.filename not in labelled]
            st.session_state.skipped_count = len(self.examples) - len(filtered)
            st.session_state.current_index = self._get_true_current_index()
            self.examples = filtered

            if not self.examples:
                st.warning(
                    "No images to label!"
                    if not st.session_state.skip_labelled
                    else "All images are already labeled!"
                )
                st.stop()

    def run(self):
        if not hasattr(self, "examples") or not self.examples:
            st.error("No valid examples found.")
            return

        example = self.examples[st.session_state.current_index]

        self._show_progress()

        with st.container(height=525):
            self._display_example(example)
        self._input_caption(example)

    def _show_progress(self):
        total = len(self.examples) + st.session_state.skipped_count
        current = st.session_state.current_index
        st.sidebar.progress(current / total)
        st.sidebar.write(
            f"{current}/{total} (Skipped: {st.session_state.skipped_count})"
        )

    def _get_true_current_index(self):
        return st.session_state.current_index + st.session_state.skipped_count

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
                    auto_captions = AutoCaption(example).create_captions()
                    for cap in auto_captions:
                        self.caption_mgr.save_caption(example, cap)

                st.session_state.current_index += 1
                if st.session_state.current_index >= len(self.examples):
                    st.success("Labelling complete!")
                    st.session_state.show_browser = False
                st.rerun()


if __name__ == "__main__":
    CaptioningApp()
