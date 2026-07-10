from langchain_core.messages import trim_messages

from utils.logger import get_logger

_logs = get_logger(__name__)

# Keep the most recent slice of the conversation so the prompt does not grow without
# bound on long sessions. We count messages (token_counter=len) rather than true tokens:
# it is dependency-free and can never miscount the model's tokenizer. The SystemMessage is
# injected separately in call_model, so it is excluded here. History rebuilt from Gradio
# contains only Human/AI messages (no ToolMessages), so trimming cannot orphan a tool call.
_MAX_MESSAGES = 12  # ~ last 6 exchanges


def trim_history(messages, max_messages: int = _MAX_MESSAGES):
    try:
        return trim_messages(
            messages,
            strategy="last",
            token_counter=len,
            max_tokens=max_messages,
            start_on="human",
            include_system=False,
        )
    except Exception as e:
        _logs.debug(f"trim_messages fallback: {e}")
        return messages[-max_messages:]
