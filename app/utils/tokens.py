import tiktoken
from typing import List
from app.schemas import ChatTurn
from app.config import settings

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def truncate_history(history: List[ChatTurn], max_turns: int = settings.MAX_HISTORY_TURNS) -> List[ChatTurn]:
    """Keeps only the most recent N turns of history."""
    if not history:
        return []
    return history[-max_turns:]