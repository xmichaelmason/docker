from transformers import PreTrainedModel, TFPreTrainedModel, PreTrainedTokenizer, ModelCard
from typing import Any, Optional

from src.domain.constants.task_type import TaskType

class ConversationalPipelineConfig:
    def __init__(self, model: Optional[Any] = None, tokenizer: Optional[Any] = None,
                 modelcard: Optional[str or Any] = None, framework: Optional[str] = None,
                 task: TaskType = TaskType.conversational, num_workers: Optional[int] = 8,
                 batch_size: Optional[int] = 1, device: Optional[int] = 0,
                 binary_output: Optional[bool] = False, min_length_for_response: Optional[int] = 32,
                 minimum_tokens: Optional[int] = 10):
        self.model = model
        self.tokenizer = tokenizer
        self.modelcard = modelcard
        self.framework = framework
        self.task = task
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.device = device
        self.binary_output = binary_output
        self.min_length_for_response = min_length_for_response
        self.minimum_tokens = minimum_tokens