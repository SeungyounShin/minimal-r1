"""
Reference Model Server with tensor parallel support.

This script launches a reference model server that can utilize multiple GPUs for tensor parallelism.
It provides an HTTP endpoint for model inference through FastAPI.

Usage example:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/launch_reference.py \
        --model_path /path/to/model \
        --port 8001 \
        --tensor_parallel_size 4
"""

import argparse
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import os

def setup_model_parallel() -> tuple:
    """设置模型并行环境"""
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    return local_rank, world_size

def load_model_in_parallel(model_path: str, tensor_parallel_size: int):
    """使用张量并行加载模型"""
    # 初始化分布式环境
    local_rank, world_size = setup_model_parallel()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 使用张量并行加载模型
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Launch Reference Model Server with tensor parallel support.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--port", type=int, required=True, help="Port to run the server on")
    parser.add_argument("--tensor_parallel_size", type=int, default=8,
                      help="Number of GPUs to use for tensor parallelism")
    args = parser.parse_args()

    print(f"Initializing Reference Model with tensor parallel size: {args.tensor_parallel_size}")
    
    # 设置环境变量
    os.environ["WORLD_SIZE"] = str(args.tensor_parallel_size)
    
    # 加载模型
    model, tokenizer = load_model_in_parallel(args.model_path, args.tensor_parallel_size)
    
    # FastAPI应用
    app = FastAPI(title="Reference Model Server", version="0.1")
    
    class InferenceRequest(BaseModel):
        prompts: List[str]
        max_tokens: int = 128
        
    class InferenceResponse(BaseModel):
        generations: List[str]
        
    @app.post("/generate", response_model=InferenceResponse)
    async def generate(request: InferenceRequest):
        """使用参考模型生成回复"""
        try:
            # 编码输入
            inputs = tokenizer(
                request.prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # 解码输出
            generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            return InferenceResponse(generations=generations)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    @app.get("/health")
    def health_check():
        """健康检查端点"""
        return {"status": "healthy"}
        
    # 启动服务
    print(f"Launching Reference Model server on port {args.port} with {args.tensor_parallel_size} GPUs...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
