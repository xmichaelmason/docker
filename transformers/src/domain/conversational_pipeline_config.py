from pydantic import BaseModel
from src.domain.constants.task_type import TaskType
from typing import Any, Optional


class ConversationalPipelineConfig(BaseModel):
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    modelcard: Optional[str or Any] = "meta-llama/Llama-2-7b-chat-hf"
    framework: Optional[str] = None
    task: TaskType = TaskType.conversational.value
    num_workers: Optional[int] = 8
    batch_size: Optional[int] = 1
    # device: Optional[int or str]
    binary_output: Optional[bool] = False
    min_length_for_response: Optional[int] = 32
    minimum_tokens: Optional[int] = 10
