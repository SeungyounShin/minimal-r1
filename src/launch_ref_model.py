#!/usr/bin/env python
# coding: utf-8

"""
ref_model_api.py
=================
- Runs `ref_model` as an API server to provide inference.
- Calculates `log probability` through the `POST /logprob` endpoint.

Example usage:
    CUDA_VISIBLE_DEVICES=1 python3 minimal_r1/launch_ref_model.py --model_name Seungyoun/Qwen2.5-7B-Open-R1-Distill
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# 설정
parser = argparse.ArgumentParser(description="Launch ref_model as an API server.")
parser.add_argument("--model_name", type=str, default="Seungyoun/Qwen2.5-7B-Open-R1-Distill", help="Model name")
parser.add_argument("--tensor_parallel_size", type=int, default=8, help="Number of GPUs to use")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 및 토크나이저 로드
print("Loading ref_model...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ref_model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
).to(DEVICE)
ref_model.eval()
print("ref_model loaded successfully!")

# FastAPI 앱 생성
app = FastAPI()

class LogProbRequest(BaseModel):
    prompts: List[str]
    generations: List[str]

@app.post("/logprob")
async def compute_logprob(request: LogProbRequest):
    """calculate log probability for given prompts and generations """
    try:
        prompts, generations = request.prompts, request.generations

        # 토큰화
        prompt_input_ids = tokenizer.batch_encode_plus(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).to(DEVICE)
        
        gen_input_ids = tokenizer.batch_encode_plus(
            generations, return_tensors="pt", padding=True, padding_side="right"
        ).to(DEVICE)

        prompt_len = prompt_input_ids.input_ids.shape[1]
        input_ids = torch.cat([prompt_input_ids.input_ids, gen_input_ids.input_ids], dim=1).long()

        with torch.inference_mode():
            logits = ref_model(input_ids, use_cache=False).logits

        logits = logits[:, :-1, :]  # (B, L-1, V)
        input_ids = input_ids[:, 1:]  # (B, L-1)

        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)

        per_token_logps = torch.stack(per_token_logps)
        per_token_logps = per_token_logps[:, prompt_len -1 :].cpu().tolist()

        return {"logprobs": per_token_logps}

    except Exception as e:
        print(f"Error computing logprob: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
