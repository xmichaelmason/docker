from typing import List, Optional
from pydantic import BaseModel

from src.domain.message import Message
from src.domain.constants.role_type import RoleType

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Message = Message(role=RoleType.assistant, content="")
    logprobs: Optional[dict]
    finish_reason: Optional[str] = None

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str = "1"
    object: str = "chat.completion"
    created: str = "2022-11-16T17:46:01-05:00"
    model: str = "gpt-3.5-turbo"
    system_fingerprint: str = "123"
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage = ChatCompletionUsage()
