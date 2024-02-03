Pipelines are made of:

        - A [tokenizer](tokenizer) in charge of mapping raw textual input to token.
        - A [model](model) to make predictions from the inputs.
        - Some (optional) post processing for enhancing model's output.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"audio-classification"`: will return a [`AudioClassificationPipeline`].
            - `"automatic-speech-recognition"`: will return a [`AutomaticSpeechRecognitionPipeline`].
            - `"conversational"`: will return a [`ConversationalPipeline`].
            - `"depth-estimation"`: will return a [`DepthEstimationPipeline`].
            - `"document-question-answering"`: will return a [`DocumentQuestionAnsweringPipeline`].
            - `"feature-extraction"`: will return a [`FeatureExtractionPipeline`].
            - `"fill-mask"`: will return a [`FillMaskPipeline`]:.
            - `"image-classification"`: will return a [`ImageClassificationPipeline`].
            - `"image-segmentation"`: will return a [`ImageSegmentationPipeline`].
            - `"image-to-image"`: will return a [`ImageToImagePipeline`].
            - `"image-to-text"`: will return a [`ImageToTextPipeline`].
            - `"mask-generation"`: will return a [`MaskGenerationPipeline`].
            - `"object-detection"`: will return a [`ObjectDetectionPipeline`].
            - `"question-answering"`: will return a [`QuestionAnsweringPipeline`].
            - `"summarization"`: will return a [`SummarizationPipeline`].
            - `"table-question-answering"`: will return a [`TableQuestionAnsweringPipeline`].
            - `"text2text-generation"`: will return a [`Text2TextGenerationPipeline`].
            - `"text-classification"` (alias `"sentiment-analysis"` available): will return a
              [`TextClassificationPipeline`].
            - `"text-generation"`: will return a [`TextGenerationPipeline`]:.
            - `"text-to-audio"` (alias `"text-to-speech"` available): will return a [`TextToAudioPipeline`]:.
            - `"token-classification"` (alias `"ner"` available): will return a [`TokenClassificationPipeline`].
            - `"translation"`: will return a [`TranslationPipeline`].
            - `"translation_xx_to_yy"`: will return a [`TranslationPipeline`].
            - `"video-classification"`: will return a [`VideoClassificationPipeline`].
            - `"visual-question-answering"`: will return a [`VisualQuestionAnsweringPipeline`].
            - `"zero-shot-classification"`: will return a [`ZeroShotClassificationPipeline`].
            - `"zero-shot-image-classification"`: will return a [`ZeroShotImageClassificationPipeline`].
            - `"zero-shot-audio-classification"`: will return a [`ZeroShotAudioClassificationPipeline`].
            - `"zero-shot-object-detection"`: will return a [`ZeroShotObjectDetectionPipeline`].

        model (`str` or [`PreTrainedModel`] or [`TFPreTrainedModel`], *optional*):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a pretrained model inheriting from [`PreTrainedModel`] (for PyTorch) or
            [`TFPreTrainedModel`] (for TensorFlow).

            If not provided, the default for the `task` will be loaded.
        config (`str` or [`PretrainedConfig`], *optional*):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from [`PretrainedConfig`].

            If not provided, the default configuration file for the requested model will be used. That means that if
            `model` is given, its default configuration will be used. However, if `model` is not supplied, this
            `task`'s default model's config is used instead.
        tokenizer (`str` or [`PreTrainedTokenizer`], *optional*):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from [`PreTrainedTokenizer`].

            If not provided, the default tokenizer for the given `model` will be loaded (if it is a string). If `model`
            is not specified or not a string, then the default tokenizer for `config` is loaded (if it is a string).
            However, if `config` is also not given or not a string, then the default tokenizer for the given `task`
            will be loaded.
        feature_extractor (`str` or [`PreTrainedFeatureExtractor`], *optional*):
            The feature extractor that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained feature extractor inheriting from [`PreTrainedFeatureExtractor`].

            Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal
            models. Multi-modal models will also require a tokenizer to be passed.

            If not provided, the default feature extractor for the given `model` will be loaded (if it is a string). If
            `model` is not specified or not a string, then the default feature extractor for `config` is loaded (if it
            is a string). However, if `config` is also not given or not a string, then the default feature extractor
            for the given `task` will be loaded.
        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
            provided.
        revision (`str`, *optional*, defaults to `"main"`):
            When passing a task name or a string model identifier: The specific model version to use. It can be a
            branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so `revision` can be any identifier allowed by git.
        use_fast (`bool`, *optional*, defaults to `True`):
            Whether or not to use a Fast tokenizer if possible (a [`PreTrainedTokenizerFast`]).
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        device (`int` or `str` or `torch.device`):
            Defines the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank like `1`) on which this
            pipeline will be allocated.
        device_map (`str` or `Dict[str, Union[int, str, torch.device]`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut). When `accelerate` library is present, set
            `device_map="auto"` to compute the most optimized `device_map` automatically (see
            [here](https://huggingface.co/docs/accelerate/main/en/package_reference/big_modeling#accelerate.cpu_offload)
            for more information).

            <Tip warning={true}>

            Do not use `device_map` AND `device` at the same time as they will conflict

            </Tip>

        torch_dtype (`str` or `torch.dtype`, *optional*):
            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`).
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether or not to allow for custom code defined on the Hub in their own modeling, configuration,
            tokenization or even pipeline files. This option should only be set to `True` for repositories you trust
            and in which you have read the code, as it will execute code present on the Hub on your local machine.
        model_kwargs (`Dict[str, Any]`, *optional*):
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).

    Returns:
        [`Pipeline`]: A suitable pipeline for the task.

    Examples:

    ```python
    >>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

    >>> # Sentiment analysis pipeline
    >>> analyzer = pipeline("sentiment-analysis")

    >>> # Question answering pipeline, specifying the checkpoint identifier
    >>> oracle = pipeline(
    ...     "question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="bert-base-cased"
    ... )

    >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
    >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    >>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
    ```"""