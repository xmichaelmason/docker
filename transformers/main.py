from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from threading import Thread
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

import logging, os

from src.domain.model_config import ModelConfig
from src.domain.message import Message
from src.domain.constants.role_type import RoleType
from src.domain.responses.chat_completion_response import ChatCompletionChoice, ChatCompletionResponse
from src.domain.requests.chat_request import ChatRequest

logging.basicConfig(
    level=logging.INFO,  # Set the logging level according to your needs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output to stdout
        # Add other handlers as needed, e.g., logging.FileHandler("app.log")
    ]
)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing, you should restrict this in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Download and load model and tokenizer
hf_token = os.getenv("HF_TOKEN")
hf_model = os.getenv("HF_MODEL")
# Now you can access your environment variables
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT")
LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT")

# Convert the strings to boolean values if needed
LOAD_IN_4BIT = LOAD_IN_4BIT.lower() == "true"
LOAD_IN_8BIT = LOAD_IN_8BIT.lower() == "true"


# Create an instance of ModelConfig with the desired parameters
model_params = ModelConfig(
    load_in_8bit=LOAD_IN_8BIT,
    load_in_4bit=LOAD_IN_4BIT,
    # llm_int8_threshold=None,
    # llm_int8_skip_modules=None,
    # llm_int8_enable_fp32_cpu_offload=False,
    # llm_int8_has_fp16_weight=False,
    # bnb_4bit_compute_dtype=torch.float16,  # Corrected to 'float16'
    # bnb_4bit_quant_type=None,
    # bnb_4bit_use_double_quant=False,
)



# Use **model_params to pass all parameters to the from_pretrained method
model = AutoModelForCausalLM.from_pretrained(
    hf_model,
    device_map="auto",
    **model_params.model_dump(),
    max_length=4096,
)

tokenizer = AutoTokenizer.from_pretrained(hf_model, token=hf_token, pad=True, max_length=4096, truncation=False)
tokenizer.add_special_tokens(
    {
        
        "pad_token": "<PAD>",
    }
)

MAX_INPUT_TOKEN_LENGTH = 512  # Define your maximum input token length here


def generate_response(message, chat_history, max_new_tokens, temperature, top_p, top_k, repetition_penalty):
    conversation = []
    
    for message in chat_history:
        conversation.append(message)

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=repetition_penalty,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
    return "".join(outputs)


messages = []

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatRequest):
    try:
        # Append the user message to the chat history
        user_message = Message(role=RoleType.user, content=request.message.content)
        messages.append(user_message)

        request.chat_history = messages
        completion = generate_response(request.message, request.chat_history,
                                       request.max_new_tokens, request.temperature, request.top_p, request.top_k,
                                       request.repetition_penalty)
        
        # Append the assistant message to the chat history
        assistant_message = Message(role=RoleType.assistant, content=completion)
        messages.append(assistant_message)

        choices = []
        for message in messages:
            choices.append(ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason="stop",
                logprobs=None,
            ))
        response = ChatCompletionResponse(
            choices=choices
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))