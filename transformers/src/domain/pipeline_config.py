from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, Any

class PipelineConfig(BaseModel):
    task: Optional[str] = Field(
        None,
        description="The task for which the pipeline will be used. If not provided, the default for the task will be loaded."
    )
    model: Optional[Union[str, Any]] = Field(
        None,
        description="The model that will be used by the pipeline to make predictions. This can be a model identifier or an actual instance of a pretrained model inheriting from PreTrainedModel (for PyTorch) or TFPreTrainedModel (for TensorFlow). If not provided, the default for the task will be loaded."
    )
    config: Optional[Union[str, Any]] = Field(
        None,
        description="The configuration that will be used by the pipeline to instantiate the model. This can be a model identifier or an actual pretrained model configuration inheriting from PretrainedConfig. If not provided, the default configuration file for the requested model will be used. That means that if model is given, its default configuration will be used. However, if model is not supplied, this task’s default model’s config is used instead."
    )
    tokenizer: Optional[Union[str, Any]] = Field(
        None,
        description="The tokenizer that will be used by the pipeline to encode data for the model. This can be a model identifier or an actual pretrained tokenizer inheriting from PreTrainedTokenizer. If not provided, the default tokenizer for the given model will be loaded (if it is a string). If model is not specified or not a string, then the default tokenizer for config is loaded (if it is a string). However, if config is also not given or not a string, then the default tokenizer for the given task will be loaded."
    )
    feature_extractor: Optional[Union[str, Any]] = Field(
        None,
        description="The feature extractor that will be used by the pipeline to encode data for the model. This can be a model identifier or an actual pretrained feature extractor inheriting from PreTrainedFeatureExtractor. Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal models. Multi-modal models will also require a tokenizer to be passed. If not provided, the default feature extractor for the given model will be loaded (if it is a string). If model is not specified or not a string, then the default feature extractor for config is loaded (if it is a string). However, if config is also not given or not a string, then the default feature extractor for the given task will be loaded."
    )
    framework: Optional[str] = Field(
        None,
        description="The framework to use, either 'pt' for PyTorch or 'tf' for TensorFlow. The specified framework must be installed. If no framework is specified, will default to the one currently installed. If no framework is specified and both frameworks are installed, will default to the framework of the model, or to PyTorch if no model is provided."
    )
    revision: Optional[str] = Field(
        "main",
        description="When passing a task name or a string model identifier: The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so revision can be any identifier allowed by git."
    )
    use_fast: bool = Field(
        True,
        description="Whether or not to use a Fast tokenizer if possible (a PreTrainedTokenizerFast)."
    )
    token: Optional[Union[str, bool]] = None
    device: Optional[Union[int, str, Any]] = None
    device_map: Optional[Any] = None
    torch_dtype: Optional[Any] = None
    trust_remote_code: Optional[bool] = None
    extra_model_kwargs: Optional[Dict[str, Any]] = None
    pipeline_class: Optional[Any] = None

    class Config:
        extra = 'allow'
