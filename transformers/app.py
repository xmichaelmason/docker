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
    TextIteratorStreamer,
    pipeline,
)

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
    hf_model, device_map="auto", trust_remote_code=True, load_in_4bit=LOAD_IN_4BIT, load_in_8bit=LOAD_IN_8BIT, token=hf_token
)
tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True, token=hf_token)

# Text generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    device_map="auto",
)

# Initial chat history (empty)
initial_chat_history = []

# Function to load initial chat history from file
def load_initial_chat_history():
    global initial_chat_history
    try:
        with open("initial_chat_history.json", "r") as file:
            initial_chat_history = json.load(file)
    except FileNotFoundError:
        # File not found, create it and populate with default message
        initial_chat_history = [
            ["Hello!", "Welcome to the chat. How can I assist you?"]
        ]
        with open("initial_chat_history.json", "w") as file:
            json.dump(initial_chat_history, file)
    except json.JSONDecodeError:
        # Error decoding JSON, use empty list as initial chat history
        initial_chat_history = []

# Function to save chat history to file
def save_chat_history(chat_history):
    with open("chat_history.json", "w") as file:
        json.dump(chat_history, file)

# Function that accepts a prompt and generates text using the llm pipeline
def generate(message, chat_history, max_new_tokens):
    instruction = "You are a helpful assistant to 'User'. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    final_prompt = f"Instruction: {instruction}\n"

    for sent, received in chat_history:
        final_prompt += "User: " + sent + "\n"
        final_prompt += "Assistant: " + received + "\n"

    final_prompt += "User: " + message + "\n"
    final_prompt += "Output:"

    # Streamer
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300.0
    )
    thread = Thread(
        target=llm_pipeline,
        kwargs={
            "text_inputs": final_prompt,
            "max_new_tokens": max_new_tokens,
            "streamer": streamer,
        },
    )
    thread.start()

    generated_text = ""
    for word in streamer:
        generated_text += word
        response = generated_text.strip()

        if "User:" in response:
            response = response.split("User:")[0].strip()
        elif "Assistant:" in response:
            response = response.split("Assistant:")[1].strip()
        else:
            response = response.strip()  # No "User:" or "Assistant:" prefix, use the entire response

        yield response

class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int

class ChatResponse(BaseModel):
    response: str

@app.on_event("startup")
async def startup_event():
    load_initial_chat_history()

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    global initial_chat_history
    chat_history = initial_chat_history.copy()
    response_generator = generate(chat_request.message, chat_history, chat_request.max_new_tokens)
    final_response = ""
    for generated_response in response_generator:
        final_response = generated_response  # Keep updating the final response with each iteration
    chat_history.append([chat_request.message, final_response])  # Append only the final response to chat history
    save_chat_history(chat_history)
    initial_chat_history = chat_history  # Update the global chat history
    return ChatResponse(response=final_response)

@app.get("/v1/chat/history", response_model=List[List[str]])
async def get_chat_history():
    global initial_chat_history
    return initial_chat_history
