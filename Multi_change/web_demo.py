import copy
import os
from io import BytesIO

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from lagent.actions import (
    ActionExecutor,
    GoogleScholar,
    GoogleSearch,
    PythonInterpreter,
    Visual_Change_Process_PythonInterpreter,
)
from lagent.agents.react import ReAct
from lagent.llms import GPTAPI
from lagent.llms.huggingface import HFTransformerCasualLM
from PIL import Image, ImageDraw
from predict import DATASET_CONFIGS, Change_Perception
from streamlit.logger import get_logger
from streamlit_image_coordinates import streamlit_image_coordinates as sic

load_dotenv()

# os.environ["http_proxy"] = "http://localhost:7890"
# os.environ["https_proxy"] = "http://localhost:7890"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,huggingface.co,openai.com"


class SessionState:

    def init_state(self):
        """Initialize session state variables."""
        st.session_state["assistant"] = []
        st.session_state["user"] = []
        st.session_state["history"] = []
        st.session_state["dataset_selected"] = "Forest-Change"

        action_list = [
            Visual_Change_Process_PythonInterpreter(),
            GoogleSearch(api_key=os.environ.get("SERPER_API_KEY")),
            GoogleScholar(api_key=os.environ.get("SERPER_API_KEY")),
        ]
        st.session_state["plugin_map"] = {action.name: action for action in action_list}
        st.session_state["model_map"] = {}
        st.session_state["model_selected"] = None
        st.session_state["plugin_actions"] = set()
        st.session_state["change_perception_instance"] = None

    def clear_state(self):
        """Clear the existing session state."""
        st.session_state["assistant"] = []
        st.session_state["user"] = []
        st.session_state["history"] = []
        st.session_state["model_selected"] = None
        if "chatbot" in st.session_state:
            st.session_state["chatbot"]._session_history = []


class StreamlitUI:

    def __init__(self, session_state: SessionState):
        self.init_streamlit()
        self.session_state = session_state

    def init_streamlit(self):
        """Initialize Streamlit's UI settings."""
        st.set_page_config(
            layout="wide",
            page_title="Forest-ChatAgent-web",
            page_icon="./docs/imgs/lagent_icon.png",
        )
        st.header("🌍 :blue[Forest-Chat] Agent ", divider="rainbow")

        st.sidebar.title("Configuration")

    def setup_sidebar(self):
        """Setup the sidebar with compact spacing and clear visual hierarchy."""

        st.sidebar.markdown("**📸 Image Upload**")

        uploaded_file_A = st.sidebar.file_uploader(
            "Image A",
            type=["png", "jpg", "jpeg"],
            key="image_A",
            label_visibility="collapsed",
        )
        uploaded_file_B = st.sidebar.file_uploader(
            "Image B",
            type=["png", "jpg", "jpeg"],
            key="image_B",
            label_visibility="collapsed",
        )

        if (
            uploaded_file_A
            and uploaded_file_B
            and uploaded_file_A.name == uploaded_file_B.name
        ):
            base, ext = os.path.splitext(uploaded_file_A.name)
            new_name_A = f"{base}_A{ext}"
            new_name_B = f"{base}_B{ext}"
        else:
            new_name_A = uploaded_file_A.name if uploaded_file_A else None
            new_name_B = uploaded_file_B.name if uploaded_file_B else None

        if uploaded_file_A:
            st.session_state["image_A_bytes"] = uploaded_file_A.read()
            st.session_state["image_A_name"] = new_name_A

        if uploaded_file_B:
            st.session_state["image_B_bytes"] = uploaded_file_B.read()
            st.session_state["image_B_name"] = new_name_B

        st.sidebar.markdown("**📊 Dataset**")

        dataset_name = st.sidebar.selectbox(
            "Dataset",
            options=list(DATASET_CONFIGS.keys()),
            index=list(DATASET_CONFIGS.keys()).index(
                st.session_state.get("dataset_selected", "Forest-Change")
            ),
            label_visibility="collapsed",
        )

        # Dataset details — visually nested
        if dataset_name in DATASET_CONFIGS:
            config = DATASET_CONFIGS[dataset_name]
            with st.sidebar.expander("↳ Dataset details", expanded=False):
                st.caption(f"Classes: {config['num_classes']}")
                st.caption(f"Pixel area: {config['pixel_area']} m²")
                st.caption(f"Model: {os.path.basename(config['checkpoint'])}")

        if dataset_name != st.session_state.get("dataset_selected"):
            st.session_state["dataset_selected"] = dataset_name
            st.session_state["change_perception_instance"] = None
            st.toast(f"Dataset changed to {dataset_name}", icon="📊")

        st.sidebar.markdown("**🤖 Language Model**")

        model_name = st.sidebar.selectbox(
            "Model",
            options=["gpt-4o-mini", "internlm-2.5-7B"],
            label_visibility="collapsed",
        )

        if model_name != st.session_state["model_selected"]:
            model = self.init_model(model_name)
            self.session_state.clear_state()
            st.session_state["model_selected"] = model_name
            st.session_state.pop("chatbot", None)
        else:
            model = st.session_state["model_map"][model_name]

        st.sidebar.markdown("**🔧 Tools**")

        plugin_name = st.sidebar.multiselect(
            "Tools",
            options=list(st.session_state["plugin_map"].keys()),
            default=list(st.session_state["plugin_map"].keys()),
            label_visibility="collapsed",
        )

        plugin_action = [st.session_state["plugin_map"][name] for name in plugin_name]

        if "chatbot" in st.session_state:
            st.session_state["chatbot"]._action_executor = ActionExecutor(
                actions=plugin_action
            )

        st.sidebar.button("🗑️ Clear conversation")

        return (
            model_name,
            model,
            plugin_action,
            uploaded_file_A,
            uploaded_file_B,
            dataset_name,
        )

    def init_model(self, option):
        """Initialize the model based on the selected option."""
        if option not in st.session_state["model_map"]:
            if option.startswith("gpt"):
                st.session_state["model_map"][option] = GPTAPI(
                    model_type=option,
                    key=os.environ.get("OPEN_AI_KEY"),
                    proxies={},
                )
            else:
                st.session_state["model_map"][option] = HFTransformerCasualLM(
                    "internlm/internlm2_5-7b-chat"
                )
        return st.session_state["model_map"][option]

    def get_change_perception(self, dataset_name):
        """Get or create Change_Perception instance for the selected dataset."""
        if (
            st.session_state["change_perception_instance"] is None
            or st.session_state["change_perception_instance"].dataset_name
            != dataset_name
        ):
            st.session_state["change_perception_instance"] = Change_Perception(
                dataset_name=dataset_name
            )
        return st.session_state["change_perception_instance"]

    def initialize_chatbot(self, model, plugin_action):
        """Initialize the chatbot with the given model and plugin actions."""
        return ReAct(llm=model, action_executor=ActionExecutor(actions=plugin_action))

    def render_user(self, prompt: str):
        with st.chat_message("user"):
            st.markdown(prompt)

    def is_valid_text(self, text):
        """Check if text is valid (not empty, not 'undefined', etc.)"""
        if not isinstance(text, str):
            return False

        cleaned = text.strip()
        # Expanded list of invalid values
        invalid_values = ["undefined", "none", "null", "", "nan", "n/a", "```"]

        return cleaned and cleaned.lower() not in invalid_values

    def clean_markdown_artifacts(self, text):
        """Remove stray markdown code fences and clean up the text."""
        if not isinstance(text, str):
            return text

        # Remove leading/trailing backticks and code fences
        cleaned = text.strip()

        # Remove trailing ``` or ```python etc
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        # Remove leading ``` or ```python etc
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].strip().startswith("```"):
                cleaned = "\n".join(lines[1:]).strip()

        return cleaned

    def render_assistant(self, agent_return):
        with st.chat_message("assistant"):
            for action in agent_return.actions:
                if action:
                    self.render_action(action)

            # Only render final response if it's valid
            if agent_return.response and self.is_valid_text(agent_return.response):
                # Strip any trailing/leading backticks before rendering
                cleaned = agent_return.response.strip().rstrip("`").lstrip("`").strip()
                if cleaned:  # Make sure there's still content after stripping
                    st.markdown(cleaned)

    def render_point_selector_tab(self, dataset_name):
        MAX_POINTS = 3

        available_images = []
        if "image_A_bytes" in st.session_state:
            available_images.append("Image A")
        if "image_B_bytes" in st.session_state:
            available_images.append("Image B")

        if (
            "image_A_bytes" not in st.session_state
            or "image_B_bytes" not in st.session_state
        ):
            st.warning(
                "Please upload both Image A and Image B from the sidebar before proceeding."
            )
            st.stop()

        with st.expander("Points of Interest", expanded=True):
            st.subheader("📍 Select Points of Interest")
            selected_image_key = st.selectbox(
                "Choose image to annotate:", available_images
            )

            if "selected_points" not in st.session_state:
                st.session_state.selected_points = {"Image A": [], "Image B": []}
            if "last_coords" not in st.session_state:
                st.session_state.last_coords = {"Image A": None, "Image B": None}

            points = st.session_state.selected_points[selected_image_key]
            last_coords = st.session_state.last_coords[selected_image_key]

            image_bytes = (
                st.session_state["image_A_bytes"]
                if selected_image_key == "Image A"
                else st.session_state["image_B_bytes"]
            )
            original_image = Image.open(BytesIO(image_bytes)).convert("RGB")

            image_to_display = original_image.copy()
            draw = ImageDraw.Draw(image_to_display)
            for x, y, _ in points:
                r = 2
                draw.ellipse((x - r, y - r, x + r, y + r), fill="red")

            left_col, right_col = st.columns([2, 1])

            with left_col:
                b1, b2 = st.columns(2)
                if b1.button("↩️ Undo Last Point"):
                    if points:
                        points.pop()
                    st.session_state.last_coords[selected_image_key] = None
                    st.rerun()

                if b2.button("🧹 Reset Points"):
                    st.session_state.selected_points[selected_image_key] = []
                    st.session_state.last_coords[selected_image_key] = None
                    st.rerun()

                st.markdown(
                    f"### 👆 Click to select up to {MAX_POINTS} points to aid segmentation (Optional)"
                )
                coords = sic(image_to_display, key=f"{selected_image_key}_click")

                t_value = 1 if selected_image_key == "Image A" else 2
                if coords and coords != last_coords:
                    if len(points) < MAX_POINTS:
                        points.append(np.array((coords["x"], coords["y"], t_value)))
                        st.session_state.last_coords[selected_image_key] = coords
                        st.rerun()
                    else:
                        st.warning(f"Max {MAX_POINTS} points allowed.")

            with right_col:
                st.markdown(f"### ✅ Selected Points ({len(points)}/{MAX_POINTS})")
                if points:
                    for i, (x, y, t) in enumerate(points):
                        st.markdown(f"**{i+1}.** (x: {x}, y: {y}, at time: {t})")
                else:
                    st.write("No points selected yet.")

        st.markdown("---")
        st.markdown(f"## 🎯 Run FC-Zero-shot on **{dataset_name}**")

        col_run, col_opts = st.columns([1, 4])

        with col_run:
            run_clicked = st.button("Run")

        with col_opts:
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                show_percentage = st.checkbox("Change %", value=True)

            with c2:
                show_patches = st.checkbox("Patch Stats", value=True)

            with c3:
                show_edges = st.checkbox("Edge/Core Stats")

            with c4:
                show_linearity = st.checkbox("Linearity Stats")

        if run_clicked:
            points = st.session_state.selected_points[selected_image_key]
            path_A = os.path.join(root_dir, st.session_state["image_A_name"])
            path_B = os.path.join(root_dir, st.session_state["image_B_name"])

            if not os.path.exists(path_A):
                with open(path_A, "wb") as f:
                    f.write(st.session_state["image_A_bytes"])

            if not os.path.exists(path_B):
                with open(path_B, "wb") as f:
                    f.write(st.session_state["image_B_bytes"])

            savepath_mask = os.path.join(root_dir, "anychange_change_mask.png")

            try:
                with st.spinner(
                    "Running FC-Zero-shot... This may take up to a minute."
                ):
                    change_perception = self.get_change_perception(dataset_name)

                    if len(points) == 0:
                        mask = change_perception.anychange_change_detection(
                            path_A=path_A,
                            path_B=path_B,
                            savepath_mask=savepath_mask,
                        )
                    else:
                        mask = change_perception.anychange_change_detection_points_of_interest(
                            path_A=path_A,
                            path_B=path_B,
                            savepath_mask=savepath_mask,
                            xyts=points,
                        )

                st.success("✅ Segmentation complete!")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(
                        path_A,
                        caption="Image A",
                        use_container_width=True,
                    )

                with col2:
                    st.image(
                        path_B,
                        caption="Image B",
                        use_container_width=True,
                    )

                with col3:
                    st.image(
                        savepath_mask,
                        caption="📍 FC-Zero-shot Output",
                        use_container_width=True,
                    )

                if show_percentage:
                    percentage_str = change_perception.compute_change_percentage(mask)
                    st.markdown(f"**Change percentage:**\n\n {percentage_str}")

                if show_patches:
                    change_statistics = change_perception.compute_patch_metrics(
                        mask, "all changes", change_perception.pixel_area
                    )
                    st.markdown(f"**Change patch statistics:**\n\n {change_statistics}")

                if show_edges:
                    edge_statistics = change_perception.compute_edge_core_change(
                        mask, "all changes"
                    )
                    st.markdown(
                        f"**Edge vs Core (edge threshold set at 20% of patch size ) analysis:**\n\n {edge_statistics}"
                    )

                if show_linearity:
                    linearity_statistics = change_perception.compute_linearity_metrics(
                        mask, "all changes"
                    )
                    st.markdown(f"**Linearity analysis:**\n\n {linearity_statistics}")
                st.markdown(f"📁 Output saved at: `{savepath_mask}`")
            except Exception as e:
                st.error(f"⚠️ Error executing the model: {e}")

    def render_action(self, action):
        expand_by_default = not (action.type in ["NoAction", "FinishAction"])

        with st.expander(action.type, expanded=expand_by_default):
            # Tool
            st.markdown(f"**Tool:** {action.type}")

            # Thought
            thought = getattr(action, "thought", None)
            if self.is_valid_text(thought):
                st.markdown(f"**Thought:** {thought}")

            # Execution Content (skip for NoAction)
            if action.type != "NoAction" and isinstance(action.args, dict):
                text = action.args.get("text")
                if self.is_valid_text(text):
                    st.markdown("**Execution Content:**")
                    st.markdown(text)

            # Results - with debugging
            if action.result:
                # DEBUG: Print all results
                print(
                    f"\n🔍 Action '{action.type}' has {len(action.result)} result(s):"
                )
                for idx, res in enumerate(action.result):
                    print(f"  [{idx}] type={type(res)}, value={res}")

                self.render_action_results(action)

    def render_action_results(self, action):
        # Track what we've rendered to avoid duplicates
        rendered_something = False

        # Collect all results by type first
        text_results = []
        image_results = []
        video_results = []
        audio_results = []

        for idx, result in enumerate(action.result):
            if not isinstance(result, dict):
                print(f"  ⚠️ Skipping non-dict result at index {idx}: {result}")
                continue

            rtype = result.get("type")
            content = result.get("content")

            print(f"  📋 Result {idx}: type='{rtype}', content='{content}'")

            if rtype == "text":
                # Check if valid before adding
                if self.is_valid_text(content):
                    # Clean up markdown artifacts before rendering
                    cleaned_content = self.clean_markdown_artifacts(content)
                    if self.is_valid_text(cleaned_content):  # Re-check after cleaning
                        print(f"    ✅ Valid text, will render")
                        text_results.append(cleaned_content)
                    else:
                        print(f"    ❌ Invalid after cleaning: '{cleaned_content}'")
                else:
                    print(f"    ❌ Invalid/undefined text, skipping: '{content}'")

            elif rtype == "image" and content and self.is_valid_text(content):
                image_results.append(content)
            elif rtype == "video" and content and self.is_valid_text(content):
                video_results.append(content)
            elif rtype == "audio" and content and self.is_valid_text(content):
                audio_results.append(content)

        # Render text results
        if text_results:
            st.markdown("**Result:**")
            for text in text_results:
                # Strip any trailing/leading backticks before rendering
                cleaned = text.strip().rstrip("`").lstrip("`").strip()
                st.markdown(cleaned)
            rendered_something = True

        # Render image (only one expected)
        if image_results:
            try:
                with open(image_results[0], "rb") as f:
                    st.image(
                        f.read(), caption="Generated Image", use_container_width=True
                    )
                rendered_something = True
            except Exception as e:
                print(f"    ❌ Error rendering image: {e}")

        # Render videos
        for video_path in video_results:
            try:
                with open(video_path, "rb") as f:
                    st.video(f.read())
                rendered_something = True
            except Exception as e:
                print(f"    ❌ Error rendering video: {e}")

        # Render audio
        for audio_path in audio_results:
            try:
                with open(audio_path, "rb") as f:
                    st.audio(f.read())
                rendered_something = True
            except Exception as e:
                print(f"    ❌ Error rendering audio: {e}")

        if not rendered_something:
            print(f"  ⚠️ Nothing was rendered for this action")


def main():
    logger = get_logger(__name__)

    if "ui" not in st.session_state:
        session_state = SessionState()
        session_state.init_state()
        st.session_state["ui"] = StreamlitUI(session_state)
    else:
        st.set_page_config(
            layout="wide",
            page_title="Forest-ChatAgent-web",
            page_icon="./docs/imgs/lagent_icon.png",
        )
        st.header("🌍:blue[Forest-Chat] Agent ", divider="rainbow")

    model_name, model, plugin_action, uploaded_file_A, uploaded_file_B, dataset_name = (
        st.session_state["ui"].setup_sidebar()
    )

    if "chatbot" not in st.session_state or model != st.session_state["chatbot"]._llm:
        st.session_state["chatbot"] = st.session_state["ui"].initialize_chatbot(
            model, plugin_action
        )

    tab_selection = st.selectbox(
        "Choose a mode:", ["Forest-Chat", "FC-Zero-shot Point Querying"]
    )

    if tab_selection == "Forest-Chat":
        for prompt, agent_return in zip(
            st.session_state["user"], st.session_state["assistant"]
        ):
            st.session_state["ui"].render_user(prompt)
            st.session_state["ui"].render_assistant(agent_return)

        if user_input := st.chat_input(""):
            st.session_state["ui"].render_user(user_input)
            st.session_state["user"].append(user_input)

            prefix = f"Using dataset: {dataset_name}. "
            file_path_A = file_path_B = None

            has_images = (
                "image_A_bytes" in st.session_state
                or "image_B_bytes" in st.session_state
            )

            if has_images:
                col1, col2 = st.columns(2)

            if "image_A_bytes" in st.session_state:
                if has_images:
                    with col1:
                        st.image(
                            st.session_state["image_A_bytes"],
                            caption="Uploaded Image_A",
                        )
                file_path_A = os.path.join(root_dir, st.session_state["image_A_name"])
                if not os.path.exists(file_path_A):
                    with open(file_path_A, "wb") as f:
                        f.write(st.session_state["image_A_bytes"])
                st.write(f"File saved at: {file_path_A}")
                prefix += f"The path of the image_A: {file_path_A}. "

            if "image_B_bytes" in st.session_state:
                if has_images:
                    with col2:
                        st.image(
                            st.session_state["image_B_bytes"],
                            caption="Uploaded Image_B",
                        )
                file_path_B = os.path.join(root_dir, st.session_state["image_B_name"])
                if not os.path.exists(file_path_B):
                    with open(file_path_B, "wb") as f:
                        f.write(st.session_state["image_B_bytes"])
                st.write(f"File saved at: {file_path_B}")
                prefix += f"The path of the image_B: {file_path_B}. "

            full_input = f"{prefix}{user_input}"
            print(f"user_input: {full_input}")
            st.session_state["history"].append(dict(role="user", content=full_input))

            with st.spinner("🤔 Processing your query..."):
                agent_return = st.session_state["chatbot"].chat(
                    st.session_state["history"]
                )

            st.session_state["history"].append(
                dict(role="assistant", content=agent_return.response)
            )
            st.session_state["assistant"].append(copy.deepcopy(agent_return))
            logger.info(agent_return.inner_steps)
            st.session_state["ui"].render_assistant(agent_return)

    elif tab_selection == "FC-Zero-shot Point Querying":
        st.session_state["ui"].render_point_selector_tab(dataset_name)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.join(root_dir, "tmp_dir")
    os.makedirs(root_dir, exist_ok=True)
    main()
