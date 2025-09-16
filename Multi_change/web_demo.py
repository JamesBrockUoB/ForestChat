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
from predict import Change_Perception
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

        action_list = [
            Visual_Change_Process_PythonInterpreter(),
            GoogleSearch(api_key=os.environ.get("SERPER_API_KEY")),
            GoogleScholar(api_key=os.environ.get("SERPER_API_KEY")),
        ]
        st.session_state["plugin_map"] = {action.name: action for action in action_list}
        st.session_state["model_map"] = {}
        st.session_state["model_selected"] = None
        st.session_state["plugin_actions"] = set()

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
        st.header("🌏 :blue[Forest-Chat] Agent ", divider="rainbow")

        st.sidebar.title("Configuration")

    def setup_sidebar(self):
        """Setup the sidebar for model and plugin selection."""
        model_name = st.sidebar.selectbox(
            "**Language Model Selection:**", options=["gpt-3.5-turbo", "internlm"]
        )
        if model_name != st.session_state["model_selected"]:
            model = self.init_model(model_name)
            self.session_state.clear_state()
            st.session_state["model_selected"] = model_name
            if "chatbot" in st.session_state:
                del st.session_state["chatbot"]
        else:
            model = st.session_state["model_map"][model_name]

        plugin_name = st.sidebar.multiselect(
            "**Tool Selection:**",
            options=list(st.session_state["plugin_map"].keys()),
            # default=[list(st.session_state['plugin_map'].keys())[0]],
            default=list(st.session_state["plugin_map"].keys()),
        )

        plugin_action = [st.session_state["plugin_map"][name] for name in plugin_name]
        if "chatbot" in st.session_state:
            st.session_state["chatbot"]._action_executor = ActionExecutor(
                actions=plugin_action
            )
        if st.sidebar.button("**Clear conversation**", key="clear"):
            self.session_state.clear_state()

        uploaded_file_A = st.sidebar.file_uploader(
            "**Upload Image_A:**", type=["png", "jpg", "jpeg"], key="image_A"
        )
        uploaded_file_B = st.sidebar.file_uploader(
            "**Upload Image_B:**", type=["png", "jpg", "jpeg"], key="image_B"
        )
        # , 'mp4', 'mp3', 'wav'

        if (
            uploaded_file_A
            and uploaded_file_B
            and uploaded_file_A.name == uploaded_file_B.name
        ):
            file_A_base, file_A_ext = os.path.splitext(uploaded_file_A.name)
            file_B_base, file_B_ext = os.path.splitext(uploaded_file_B.name)
            new_name_A = f"{file_A_base}_A{file_A_ext}"
            new_name_B = f"{file_B_base}_B{file_B_ext}"
        else:
            new_name_A = uploaded_file_A.name if uploaded_file_A else None
            new_name_B = uploaded_file_B.name if uploaded_file_B else None

        if uploaded_file_A:
            image_A_bytes = uploaded_file_A.read()
            st.session_state["image_A_bytes"] = image_A_bytes
            st.session_state["image_A_name"] = new_name_A

        if uploaded_file_B:
            image_B_bytes = uploaded_file_B.read()
            st.session_state["image_B_bytes"] = image_B_bytes
            st.session_state["image_B_name"] = new_name_B

        return model_name, model, plugin_action, uploaded_file_A, uploaded_file_B

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

    def initialize_chatbot(self, model, plugin_action):
        """Initialize the chatbot with the given model and plugin actions."""
        return ReAct(llm=model, action_executor=ActionExecutor(actions=plugin_action))

    def render_user(self, prompt: str):
        with st.chat_message("user"):
            st.markdown(prompt)

    def render_assistant(self, agent_return):
        with st.chat_message("assistant"):
            for action in agent_return.actions:
                if action:
                    self.render_action(action)
            if agent_return.response:
                st.markdown(agent_return.response)

    def render_point_selector_tab(self):
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
        st.markdown("## 🎯 Run AnyChange Model")

        if st.button("Run"):
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
                    "Running AnyChange Model... This may take up to a minute."
                ):
                    change_perception = Change_Perception()
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
                        caption="🔍 AnyChange Output",
                        use_container_width=True,
                    )

                percentage_str = change_perception.compute_deforestation_percentage(
                    mask
                )
                st.markdown(percentage_str)
                st.markdown(f"📁 Output saved at: `{savepath_mask}`")
            except Exception as e:
                st.markdown(
                    f"☠️ Uh oh! Ran into a problem executing the model: {e} - double check model hyperparameters"
                )

    def render_action(self, action):
        with st.expander(action.type, expanded=True):
            st.markdown(
                "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>Tool</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                + action.type
                + "</span></p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>Thought</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                + action.thought
                + "</span></p>",
                unsafe_allow_html=True,
            )
            if isinstance(action.args, dict) and "text" in action.args:
                st.markdown(
                    "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>Execution Content</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",  # noqa E501
                    unsafe_allow_html=True,
                )
                st.markdown(action.args["text"])
            if action.result:
                self.render_action_results(action)

    def render_action_results(self, action):
        """Render the results of action, including text, images, videos, and
        audios."""
        if isinstance(action.result, dict):
            action.result = list(action.result)

        for result in action.result:
            if isinstance(result, dict):
                st.markdown(
                    "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'> Result</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",  # noqa E501
                    unsafe_allow_html=True,
                )
                if "text" in result["type"]:
                    st.markdown(
                        "<p style='text-align: left;'>" + result["content"] + "</p>",
                        unsafe_allow_html=True,
                    )
                if "image" in result["type"]:
                    image_path = result["content"]
                    image_data = open(image_path, "rb").read()
                    st.image(image_data, caption="Generated Image")
                if "video" in result["type"]:
                    video_data = result["content"]
                    video_data = open(video_data, "rb").read()
                    st.video(video_data)
                if "audio" in result["type"]:
                    audio_data = result["content"]
                    audio_data = open(audio_data, "rb").read()
                    st.audio(audio_data)


def main():
    logger = get_logger(__name__)
    # Initialize Streamlit UI and setup sidebar
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
        st.header("🌏:blue[Forest-Chat] Agent ", divider="rainbow")
    model_name, model, plugin_action, uploaded_file_A, uploaded_file_B = (
        st.session_state["ui"].setup_sidebar()
    )

    # Initialize chatbot if it is not already initialized
    # or if the model has changed
    if "chatbot" not in st.session_state or model != st.session_state["chatbot"]._llm:
        st.session_state["chatbot"] = st.session_state["ui"].initialize_chatbot(
            model, plugin_action
        )

    tab_selection = st.selectbox("Choose a mode:", ["Forest-Chat Agent", "AnyChange"])

    if tab_selection == "Forest-Chat Agent":
        for prompt, agent_return in zip(
            st.session_state["user"], st.session_state["assistant"]
        ):
            st.session_state["ui"].render_user(prompt)
            st.session_state["ui"].render_assistant(agent_return)

        if user_input := st.chat_input(""):
            st.session_state["ui"].render_user(user_input)
            st.session_state["user"].append(user_input)

            prefix = ""
            file_path_A = file_path_B = None

            if "image_A_bytes" in st.session_state:
                st.image(st.session_state["image_A_bytes"], caption="Uploaded Image_A")
                file_path_A = os.path.join(root_dir, st.session_state["image_A_name"])
                if not os.path.exists(file_path_A):
                    with open(file_path_A, "wb") as f:
                        f.write(st.session_state["image_A_bytes"])
                st.write(f"File saved at: {file_path_A}")
                prefix += f"The path of the image_A: {file_path_A}. "

            if "image_B_bytes" in st.session_state:
                st.image(st.session_state["image_B_bytes"], caption="Uploaded Image_B")
                file_path_B = os.path.join(root_dir, st.session_state["image_B_name"])
                if not os.path.exists(file_path_B):
                    with open(file_path_B, "wb") as f:
                        f.write(st.session_state["image_B_bytes"])
                st.write(f"File saved at: {file_path_B}")
                prefix += f"The path of the image_B: {file_path_B}. "

            full_input = f"{prefix}{user_input}"
            print(f"user_input: {full_input}")
            st.session_state["history"].append(dict(role="user", content=full_input))

            agent_return = st.session_state["chatbot"].chat(st.session_state["history"])
            st.session_state["history"].append(
                dict(role="assistant", content=agent_return.response)
            )
            st.session_state["assistant"].append(copy.deepcopy(agent_return))
            logger.info(agent_return.inner_steps)
            st.session_state["ui"].render_assistant(agent_return)

    elif tab_selection == "AnyChange":
        st.session_state["ui"].render_point_selector_tab()


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.join(root_dir, "tmp_dir")
    os.makedirs(root_dir, exist_ok=True)
    main()
