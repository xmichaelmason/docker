from typing import Any, Optional
from pydantic import BaseModel
import torch

class ModelConfig(BaseModel):
    load_in_8bit: Optional[bool] = None
    load_in_4bit: Optional[bool] = None
    # llm_int8_threshold: Optional[float] = None
    # llm_int8_skip_modules: Optional[str] = None
    # llm_int8_enable_fp32_cpu_offload: Optional[bool] = None
    # llm_int8_has_fp16_weight: Optional[bool] = None
    # bnb_4bit_compute_dtype: Optional[str] = torch.float16
    # bnb_4bit_quant_type: Optional[str] = 'fp4'
    # bnb_4bit_use_double_quant: Optional[bool] = False