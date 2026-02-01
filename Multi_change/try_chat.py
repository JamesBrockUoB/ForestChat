import os

from dotenv import load_dotenv
from lagent.actions import (
    ActionExecutor,
    GoogleSearch,
    PythonInterpreter,
    Visual_Change_Process_PythonInterpreter,
)
from lagent.agents import ReAct
from lagent.llms import GPTAPI, HFTransformer, HFTransformerCasualLM

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

load_dotenv()

llm = GPTAPI(
    model_type="gpt-4o-mini", key=os.environ.get("OPEN_AI_KEY")
)  # KEY from openai
# llm = HFTransformer(r'internlm/internlm-chat-7b-v1_1')
# llm = HFTransformerCasualLM(r'internlm/internlm-chat-7b-v1_1')
search_tool = GoogleSearch(
    api_key=os.environ.get("SERPER_KEY")
)  # key from google search
python_interpreter = PythonInterpreter()
imgchange_python_interpreter = Visual_Change_Process_PythonInterpreter()

chatbot = ReAct(
    llm=llm,
    action_executor=ActionExecutor(actions=[search_tool, imgchange_python_interpreter]),
)

if __name__ == "__main__":
    history = []
    while True:
        user_input = input("\n||<User>||Please input your message: ")
        # user_input='The path of picture A is <path A>, the path of picture B is <path B>, the masks for detecting changes in buildings and roads are displayed in red and green respectively. The result can be saved in <save path>'
        # user_input='The path of picture A is <path A>, the path of picture B is <path B, Describe what has changed in the two images'

        print("\n||Start thinking ...")
        history.append(dict(role="user", content=user_input))
        agent_return = chatbot.chat(history)
        history.append(dict(role="assistant", content=agent_return.response))
        print("\n||<Change-Agent>|| The response of Agent:")
        print(agent_return.response)
