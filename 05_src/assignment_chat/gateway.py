"""Route OpenAI chat and embeddings through the course API gateway when USE_GATEWAY=true.

The course env sets USE_GATEWAY=true and the direct OPENAI_API_KEY has no quota, so both
the chat model and the embedding function must go through the gateway (base URL + an
x-api-key header), matching the pattern in lab 04_7 and utils/clients.py.
"""
import os

from langchain.chat_models import init_chat_model
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

_GATEWAY_URL = "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1"


def _use_gateway() -> bool:
    return os.getenv("USE_GATEWAY", "FALSE").upper() == "TRUE"


def get_chat_model(model: str = None):
    model = model or f"openai:{os.getenv('MODEL', 'gpt-4o-mini')}"
    if _use_gateway():
        return init_chat_model(
            model,
            base_url=_GATEWAY_URL,
            api_key="any value",
            default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
        )
    return init_chat_model(model)


def get_embedding_function() -> OpenAIEmbeddingFunction:
    model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    if _use_gateway():
        return OpenAIEmbeddingFunction(
            api_key="any value",
            model_name=model_name,
            api_base=_GATEWAY_URL,
            default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
        )
    return OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=model_name,
    )
