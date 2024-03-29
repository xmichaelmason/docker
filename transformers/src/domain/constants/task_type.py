from enum import Enum

class TaskType(Enum):
    audio_classification = 'audio-classification'
    automatic_speech_recognition = 'automatic-speech-recognition'
    conversational = 'conversational'
    depth_estimation = 'depth-estimation'
    document_question_answering = 'document-question-answering'
    feature_extraction = 'feature-extraction'
    fill_mask = 'fill-mask'
    image_classification = 'image-classification'
    image_segmentation = 'image-segmentation'
    image_to_image = 'image-to-image'
    image_to_text = 'image-to-text'
    mask_generation = 'mask-generation'
    ner = 'ner'
    object_detection = 'object-detection'
    question_answering = 'question-answering'
    sentiment_analysis = 'sentiment-analysis'
    summarization = 'summarization'
    table_question_answering = 'table-question-answering'
    text_classification = 'text-classification'
    text_generation = 'text-generation'
    text_to_audio = 'text-to-audio'
    text_to_speech = 'text-to-speech'
    text2text_generation = 'text2text-generation'
    token_classification = 'token-classification'
    translation = 'translation'
    video_classification = 'video-classification'
    visual_question_answering = 'visual-question-answering'
    vqa = 'vqa'
    zero_shot_audio_classification = 'zero-shot-audio-classification'
    zero_shot_classification = 'zero-shot-classification'
    zero_shot_image_classification = 'zero-shot-image-classification'
    zero_shot_object_detection = 'zero-shot-object-detection'
    translation_XX_to_YY = 'translation_XX_to_YY'