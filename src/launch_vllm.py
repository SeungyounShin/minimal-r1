#!/usr/bin/env python
# coding: utf-8

"""
vLLM Server with tensor parallel support.

This script launches a vLLM server that can utilize multiple GPUs for tensor parallelism.
It provides an HTTP endpoint '/generate' through FastAPI (uvicorn).

Usage example:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/launch_vllm.py \
        --model_path /path/to/model \
        --port 8000 \
        --tensor_parallel_size 4
"""

import argparse
import io
import torch
import yaml
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
# vLLM
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser(description="Launch vLLM Server with a specified model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--port", type=int, required=True, help="Port to run the server on")
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="Number of GPUs to use")
    args = parser.parse_args()

    print(f"Initializing vLLM with tensor parallel size: {args.tensor_parallel_size}")
    
    # 初始化vLLM模型
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,  # 启用张量并行
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
    )

    # 基本采样参数
    default_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=128,
    )

    # FastAPI应用
    app = FastAPI(title="vLLM Server", version="0.1")

    class GenerateRequest(BaseModel):
        prompts: List[str]
        num_gen: int = 1
        temperature: float = 0.6
        max_tokens: int = 128

    class GenerateResponse(BaseModel):
        generations: List[List[str]]

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        """使用vLLM生成num_gen个样本"""
        sampling_params = SamplingParams(
            temperature=req.temperature,
            top_p=0.95,
            max_tokens=req.max_tokens,
        )

        num_prompts = len(req.prompts)
        outputs = llm.generate(req.prompts * req.num_gen, sampling_params)

        generations = [[] for _ in range(num_prompts)]
        for i, output in enumerate(outputs):
            prompt_idx = i % num_prompts
            generations[prompt_idx].append(output.outputs[0].text)

        return GenerateResponse(generations=generations)

    @app.get("/health")
    def health_check():
        """健康检查端点"""
        return {"status": "healthy"}

    @app.post("/load_weights")
    async def load_weights(request: Request):
        """加载新的模型权重"""
        if llm is None:
            raise HTTPException(status_code=400, detail="LLM is not initialized.")

        try:
            weights_data = await request.body()
            buffer = io.BytesIO(weights_data)
            state_dict = torch.load(buffer, map_location="cpu")
            llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(state_dict.items())
            print("\033[32mNew model weights loaded.\033[0m")
            return {"status": "success", "message": "Model weights loaded."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load weights: {str(e)}")

    # 启动服务
    print(f"Launching vLLM API server on port {args.port} with {args.tensor_parallel_size} GPUs...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
