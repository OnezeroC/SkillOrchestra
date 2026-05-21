#!/usr/bin/env python3
"""Minimal OpenAI-compatible server for Qwen2.5-3B using transformers only.
No SGLang/vLLM/flashinfer dependencies — just torch + transformers + fastapi.
"""
import argparse, time, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

model = None
tokenizer = None


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[dict]
    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.9


class GenerateRequest(BaseModel):
    text: str
    sampling_params: dict = {}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt = req.messages[-1]["content"] if req.messages else ""
    messages = [{"role": m["role"], "content": m["content"]} for m in req.messages]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return {
        "choices": [{"message": {"content": response_text}}],
        "usage": {"prompt_tokens": inputs.input_ids.shape[1], "completion_tokens": len(outputs[0]) - inputs.input_ids.shape[1]},
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    """SGLang-compatible /generate endpoint for call_router."""
    text = req.text
    params = req.sampling_params
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    max_tokens = params.get("max_new_tokens", 512)
    temperature = params.get("temperature", 0.6)
    top_p = params.get("top_p", 0.9)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {
        "text": response_text,
        "meta_info": {
            "prompt_tokens": inputs.input_ids.shape[1],
            "completion_tokens": len(outputs[0]) - inputs.input_ids.shape[1],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=30008)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    print(f"Model loaded on {model.device}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
