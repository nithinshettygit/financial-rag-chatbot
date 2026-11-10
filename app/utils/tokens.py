import tiktoken
from typing import Any, List, Optional, Tuple
from app.schemas import ChatTurn
from app.config import settings

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def count_message_tokens(messages: List[dict[str, Any]]) -> int:
    text = "\n".join(
        f"{message.get('role', '')}: {message.get('content', '')}" for message in messages
    )
    return count_tokens(text)


def provider_token_usage(response: Any) -> Optional[Tuple[int, int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return input_tokens, output_tokens
    return None


def log_token_usage(span: Any, response: Any, estimated_input: int, estimated_output: int) -> Tuple[int, int]:
    usage = provider_token_usage(response)
    if usage is None:
        input_tokens, output_tokens = estimated_input, estimated_output
        source = "estimate"
    else:
        input_tokens, output_tokens = usage
        source = "provider"
    span.set_attribute("input_tokens", input_tokens)
    span.set_attribute("output_tokens", output_tokens)
    span.set_attribute("total_tokens", input_tokens + output_tokens)
    span.set_attribute("llm_calls", 1)
    span.set_attribute("token_count_source", source)
    return input_tokens, output_tokens


def truncate_history(history: List[ChatTurn], max_turns: int = settings.MAX_HISTORY_TURNS) -> List[ChatTurn]:
    """Keeps only the most recent N turns of history."""
    if not history:
        return []
    return history[-max_turns:]