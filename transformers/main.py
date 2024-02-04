import os
from typing import List, Optional

from llama_index.llms import HuggingFaceInferenceAPI, HuggingFaceLLM
from llama_index import set_global_tokenizer
from transformers import AutoTokenizer

set_global_tokenizer(
    AutoTokenizer.from_pretrained(os.getenv("HF_MODEL")).encode
)

HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
locally_run = HuggingFaceLLM(model_name=os.getenv("HF_MODEL"))

completion_response = locally_run.complete("To infinity, and")
print(completion_response)