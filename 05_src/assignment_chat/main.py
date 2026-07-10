from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

from assignment_chat.prompts import return_instructions
from assignment_chat.gateway import get_chat_model
from assignment_chat.tools_reviews import recommend_albums
from assignment_chat.tools_api import get_artist_releases
from assignment_chat.tools_websearch import web_search
from utils.logger import get_logger

_logs = get_logger(__name__)
load_dotenv(".env")
load_dotenv(".secrets")

chat_agent = get_chat_model()
tools = [recommend_albums, get_artist_releases, web_search]
instructions = return_instructions()


def call_model(state: MessagesState):
    """LLM decides whether to answer or call a tool."""
    response = chat_agent.bind_tools(tools).invoke(
        [SystemMessage(content=instructions)] + state["messages"]
    )
    return {"messages": [response]}


def get_graph():
    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")
    return builder.compile()
