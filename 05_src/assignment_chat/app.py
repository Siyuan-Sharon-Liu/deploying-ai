from assignment_chat.main import get_graph
from assignment_chat.memory import trim_history
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr
from dotenv import load_dotenv

from utils.logger import get_logger

_logs = get_logger(__name__)
load_dotenv(".env")
load_dotenv(".secrets")

llm = get_graph()


def assistant_chat(message: str, history: list[dict]) -> str:
    msgs = []
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=message))
    msgs = trim_history(msgs)
    response = llm.invoke({"messages": msgs})
    return response["messages"][-1].content


chat = gr.ChatInterface(
    fn=assistant_chat,
    title="The Record Counter",
    description="Ask Vinny for album recommendations, artist facts, and music news.",
)

if __name__ == "__main__":
    _logs.info("Starting Assignment Chat App...")
    chat.launch()
