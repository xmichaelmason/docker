from typing import List, Optional
from pydantic import BaseModel

from src.domain.message import Message


class ChatRequest(BaseModel):
    message: Message
    chat_history: Optional[List[Message]]
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2


class ChatCompletion(BaseModel):
    completion: str