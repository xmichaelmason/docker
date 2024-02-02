from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from threading import Thread
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import json, logging, os

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    Conversation
)

from src.domain.conversational_pipeline_config import ConversationalPipelineConfig
from src.domain.message import Message

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

model = AutoModelForCausalLM.from_pretrained(
    hf_model, device_map="auto", load_in_4bit=LOAD_IN_4BIT, load_in_8bit=LOAD_IN_8BIT, token=hf_token, max_memory={0: "6GiB", "cpu": "32GiB"}
)
tokenizer = AutoTokenizer.from_pretrained(hf_model, token=hf_token)

chat_config = ConversationalPipelineConfig(model=model, tokenizer=tokenizer)
chatbot = pipeline(
    task=chat_config.task.value,
    model=chat_config.model,
    model_card=chat_config.modelcard,
    tokenizer=chat_config.tokenizer,
    framework=chat_config.framework,
    batch_size=chat_config.batch_size,
    num_workers=chat_config.num_workers,
    binary_output=chat_config.binary_output,
    minimum_tokens=chat_config.minimum_tokens,
    min_length_for_response=chat_config.min_length_for_response
)

conversation = Conversation()


@app.post("/v1/chat/completions", response_model=str)
async def chat_completions(user_input: str):
    global conversation
    if len(conversation) == 0:
        system_message = "You are a helpful assistant."
        conversation.add_message({"role": "system", "content": system_message})
    conversation.add_message({"role": "user", "content": user_input})
    conversation = chatbot(conversation)
    return str(conversation.messages)

@app.get("/v1/chat/history", response_model=None)
async def chat_history():
    global conversation
    return conversation