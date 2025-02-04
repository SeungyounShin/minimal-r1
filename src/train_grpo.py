#!/usr/bin/env python
# coding: utf-8

"""
train_grpo.py
=============
|   gpu0     |   gpu1     |   gpu2 ~ 7  | 
| generation | reference  |    policy   | 

Using AI-MO/NuminaMath-TIR dataset:
   - Use 'problem' key as Prompt

Example usage:
    # Assign to GPUs 2~7 (example)
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --config_file configs/zero3.yaml train_grpo.py --max_tokens 2048
"""

import requests
import io
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate.utils import broadcast_object_list, gather_object
from accelerate.utils import tqdm
import wandb
from transformers.utils import is_peft_available
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import logging
from collections import defaultdict
import yaml
import os

if is_peft_available():
    from peft import PeftConfig, get_peft_model

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from reward_fn import reward_funcs_registry

def get_generation_from_vllm(prompts, num_gen=1, temperature=0.8, max_tokens=128, vllm_server_url=None):
    """api call for generation from vLLM"""
    payload = {
        "prompts": prompts,
        "num_gen": num_gen,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    resp = requests.post(f"{vllm_server_url}/generate", json=payload, timeout=500)
    resp.raise_for_status()
    data = resp.json()
                               # <----num_gen---->
    return data["generations"] # [[gen1_1, gen1_2], [gen2_1, gen2_2], ...]

def compute_logprob_from_ref(prompt_and_gen, ref_model_api_url=None):
    """api call for log probability of reference model"""
    prompts, generations = zip(*prompt_and_gen)
    try:
        payload = {
            "prompts": prompts,
            "generations": generations
        }

        resp = requests.post(f"{ref_model_api_url}/logprob", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        return torch.tensor(data["logprobs"])  # [[logp1, logp2], [logp3, logp4], ...]
    except requests.exceptions.RequestException as e:
        print(f"Error requesting ref_model API: {e}")
        return None

def compute_logprob(model, tokenizer, prompt_and_gen):
    """copied from https://github.com/huggingface/trl/blob/249fe97158612839255468892bf74a4d823c1bc6/trl/trainer/grpo_trainer.py#L430"""
    prompts, generations = zip(*prompt_and_gen)
    prompt_input_ids = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True, padding_side="left").to('cuda')
    gen_input_ids = tokenizer.batch_encode_plus(generations, return_tensors="pt", padding=True, padding_side="right").to('cuda')

    prompt_len = prompt_input_ids.input_ids.shape[1]
    input_ids = torch.cat([prompt_input_ids.input_ids, gen_input_ids.input_ids], dim=1).long() # (B, L)
    
    logits = model(input_ids, use_cache=False).logits
    logits = logits[:, :-1, :]  # (B, L-1, V)
    input_ids = input_ids[:, 1:]  # (B, L-1)
    
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)

    per_token_logps = torch.stack(per_token_logps)
    per_token_logps = per_token_logps[:, prompt_len -1 :]

    is_eos = gen_input_ids.input_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=gen_input_ids.input_ids.device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=gen_input_ids.input_ids.device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    return per_token_logps, completion_mask

def save_with_accelerate(accelerator, model, output_dir):
    accelerator.deepspeed_plugin.zero3_save_16bit_model = True
    accelerator.deepspeed_plugin.stage3_gather_16bit_weights_on_model_save = True

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    # check if state_dict is a dict has empty tensor
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            output_dir, 
            is_main_process=accelerator.is_main_process, 
            save_function=accelerator.save, 
            state_dict=state_dict
        )

def create_reference_model(model):
    """Create a reference model by copying the model's state dict."""
    ref_model = type(model)(model.config)
    ref_model.load_state_dict(model.state_dict())
    return ref_model

def generate_model_card(
    base_model=None,
    model_name=None,
    dataset_name=None,
    tags=None,
    wandb_url=None,
    comet_url=None,
    trainer_name="GRPO",
    trainer_citation=None,
    paper_title=None,
    paper_id=None,
):
    """Generate a model card with training details."""
    content = []
    
    if model_name:
        content.append(f"# {model_name}")
    else:
        content.append("# GRPO Fine-tuned Model")
        
    content.append("\n## Model Details")
    if base_model:
        content.append(f"- Base model: {base_model}")
    if dataset_name:
        content.append(f"- Training dataset: {dataset_name}")
    
    content.append("\n## Training Details")
    content.append(f"- Training method: {trainer_name}")
    if paper_title and paper_id:
        content.append(f"- Paper: [{paper_title}](https://arxiv.org/abs/{paper_id})")
    
    if wandb_url or comet_url:
        content.append("\n## Training Logs")
        if wandb_url:
            content.append(f"- [Weights & Biases]({wandb_url})")
        if comet_url:
            content.append(f"- [Comet ML]({comet_url})")
    
    if trainer_citation:
        content.append("\n## Citation")
        content.append("```bibtex")
        content.append(trainer_citation)
        content.append("```")
    
    if tags:
        content.append("\n## Tags")
        for tag in tags:
            content.append(f"- {tag}")
    
    return "\n".join(content)

def main(args):
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取模型配置
    model_config = config['models']
    model_path = model_config['train']['path']
    
    # 处理输出路径（相对路径转绝对路径）
    output_path = os.path.join(project_root, model_config['output']['path'])
    checkpoint_dir = os.path.join(output_path, model_config['output']['checkpoint_dir'])
    log_dir = os.path.join(output_path, model_config['output']['log_dir'])
    
    # 创建必要的目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化accelerator
    accelerator = Accelerator()
    
    # 加载模型
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=getattr(torch, args.torch_dtype)
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # 加载数据集
    dataset = load_dataset(args.dataset_name, split="train")
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training parameters
    per_device_train_batch_size = args.batch_size
    epochs = args.epochs
    num_gen = args.num_gen
    max_tokens = args.max_tokens
    beta = args.beta
    reward_funcs = [
        reward_funcs_registry["accuracy"],
        reward_funcs_registry["format"],
    ]
    gradient_accumulation_steps = accelerator.deepspeed_plugin.gradient_accumulation_steps
    num_gpus = accelerator.num_processes

    # Optional PEFT configuration
    peft_config = getattr(args, "peft_config", None)
    if peft_config is not None and is_peft_available():
        policy_model = get_peft_model(policy_model, peft_config)

    policy_model.train()
    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})

    # Initialize reference model
    if is_deepspeed_zero3_enabled():
        ref_model = AutoModelForCausalLM.from_pretrained(model_path)
    elif peft_config is None:
        ref_model = create_reference_model(policy_model)
    else:
        ref_model = None

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
    )

    # Prepare for distributed training
    policy_model, train_dataloader, optimizer = accelerator.prepare(policy_model, train_dataloader, optimizer)
    if ref_model is not None:
        if accelerator.is_deepspeed_enabled:
            ref_model = prepare_deepspeed(ref_model, accelerator)
        else:
            ref_model = accelerator.prepare_model(ref_model, evaluation_mode=True)

    # Initialize vLLM if enabled
    if args.use_vllm:
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Please install it with `pip install vllm`")
        
        if accelerator.is_main_process:
            vllm_device = f"cuda:{accelerator.num_processes}"  # Use next available GPU
            try:
                llm = LLM(
                    model=model_path,
                    device=vllm_device,
                    gpu_memory_utilization=0.9,
                    enable_prefix_caching=True,
                )
                sampling_params = SamplingParams(
                    n=num_gen,
                    temperature=args.temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize vLLM: {e}")

    progress_bar = tqdm(total=len(train_dataloader) * epochs, desc=f"Training")

    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(
            project="minimal-r1",
            config={
                "learning_rate": args.lr,
                "epochs": epochs,
                "batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_gen": num_gen,
                "max_tokens": max_tokens,
                "beta": beta,
                "model_name": model_path,
                "dataset_name": args.dataset_name,
                "system_prompt": args.system_prompt,
                "peft_config": str(peft_config) if peft_config else None,
            },
        )

    global_step = 0
    _metrics = defaultdict(list)  # Store metrics for logging
    
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            _metrics["epoch"] = epoch + 1
            _metrics["iteration"] = global_step + 1

            batch_size = len(batch["problem"])
            prompts : list[str] = batch["problem"]
            prompts = [[{"role": "system", "content": args.system_prompt}, {"role": "user", "content": prompt}] for prompt in prompts]
            prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]

            all_prompts : list[str] = gather_object(prompts)

            # Generation step
            generations = [[None] * num_gen] * (batch_size * num_gpus)
            if accelerator.is_main_process:
                if args.use_vllm:
                    outputs = llm.generate(all_prompts, sampling_params=sampling_params, use_tqdm=False)
                    generations = [[out.text for out in completions.outputs] for completions in outputs]
                else:
                    generation_start_time = time.time()
                    generations = get_generation_from_vllm(all_prompts, num_gen=num_gen, max_tokens=max_tokens, vllm_server_url=args.vllm_server_url)
                    generation_end_time = time.time()
                    _metrics["generation_time"].append(generation_end_time - generation_start_time)
            
            generation_global = broadcast_object_list(generations, from_process=0)
            generations = generation_global[accelerator.process_index * batch_size : accelerator.process_index * batch_size + batch_size]
            local_generations = [item for sublist in generations for item in sublist]
            local_prompts = [p for p in prompts for _ in range(num_gen)]
            
            prompt_and_gen = [(prompt, gen) for prompt, gen in zip(local_prompts, local_generations)]

            with accelerator.accumulate():
                # Compute policy model log probabilities
                policy_logp_start_time = time.time()
                logp, completion_mask = compute_logprob(policy_model, tokenizer, prompt_and_gen)
                policy_logp_end_time = time.time()
                _metrics["policy_logp_time"].append(policy_logp_end_time - policy_logp_start_time)

                # Compute reference model log probabilities
                ref_logp_start_time = time.time()
                if ref_model is not None:
                    ref_logp = compute_logprob(ref_model, tokenizer, prompt_and_gen)[0]
                else:
                    with policy_model.disable_adapter():
                        ref_logp = compute_logprob(policy_model, tokenizer, prompt_and_gen)[0]
                ref_logp_end_time = time.time()
                _metrics["ref_logp_time"].append(ref_logp_end_time - ref_logp_start_time)
                
                ref_logp = ref_logp.to(accelerator.device)
                per_token_kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1
            
                # Compute rewards
                rewards_per_func = torch.zeros(len(prompts) * num_gen, len(reward_funcs), device=device)
                
                for i, reward_func in enumerate(reward_funcs):
                    reward_kwargs = {key: [] for key in batch.keys() if key not in ["prompt", "completion"]}
                    reward_kwargs['problem'] = [p for p in batch["problem"] for _ in range(num_gen)]
                    reward_kwargs['solution'] = [sol for sol in batch["solution"] for _ in range(num_gen)]
                    reward_kwargs['completions'] = local_generations
                    
                    rewards = reward_func(**reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(rewards, device=device)
                    _metrics[f"rewards/{reward_func.__name__}"].append(rewards_per_func[:, i].mean().item())
                
                rewards = rewards_per_func.sum(dim=1)
                _metrics["reward"].append(rewards.mean().item())
                
                # Calculate advantages
                mean_grouped_rewards = rewards.view(-1, num_gen).mean(dim=1)
                std_grouped_rewards = rewards.view(-1, num_gen).std(dim=1)
                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_gen, dim=0)
                std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_gen, dim=0)
                advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

                # Compute loss
                per_token_loss = torch.exp(logp - logp.detach()) * advantages.unsqueeze(1)
                per_token_loss = -(per_token_loss - beta * per_token_kl)
                loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

                # Optimization step
                backward_start_time = time.time()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                backward_end_time = time.time()
                _metrics["backward_time"].append(backward_end_time - backward_start_time)

            # Weight sync for vLLM
            sync_start_time = time.time()
            accelerator.wait_for_everyone()
            if (step + 1) % gradient_accumulation_steps == 0:
                full_state_dict = accelerator.get_state_dict(policy_model)

                if args.use_vllm and accelerator.is_main_process:
                    try:
                        llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(full_state_dict.items())
                    except Exception as e:
                        print(f"[ERROR] Failed to load weights to vLLM: {e}")
                else:
                    if accelerator.is_main_process:
                        buffer = io.BytesIO()
                        torch.save(full_state_dict, buffer)
                        buffer.seek(0)
                        try:
                            r = requests.post(f"{args.vllm_server_url}/load_weights", data=buffer.read(), timeout=500)
                            r.raise_for_status()
                        except requests.exceptions.RequestException as e:
                            print(f"[ERROR] Failed to load weights to vLLM: {e}")
            sync_end_time = time.time()
            _metrics["sync_time"].append(sync_end_time - sync_start_time)
            accelerator.wait_for_everyone()

            # Log metrics
            completion_length = accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
            _metrics["completion_length"].append(completion_length)

            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            _metrics["kl"].append(accelerator.gather_for_metrics(mean_kl).mean().item())
            _metrics["loss"].append(accelerator.gather_for_metrics(loss).mean().item())

            if accelerator.is_main_process and (step + 1) % args.logging_steps == 0:
                # Average the metrics
                metrics = {key: sum(val) / len(val) for key, val in _metrics.items() if isinstance(val, list)}
                wandb.log(metrics)
                _metrics.clear()

                # Log generation examples
                if hasattr(args, "log_examples") and args.log_examples:
                    wandb.log({
                        "generation_table": wandb.Table(
                            columns=["step", "prompt", "generation"],
                            data=[[global_step + 1, p, g] for p, g in zip(local_prompts[:3], local_generations[:3])]
                        )
                    })

            progress_bar.update(1)
            global_step += 1

            if (step) % args.save_step == 0:    
                 # saveing every epoch
                accelerator.wait_for_everyone()
                save_with_accelerate(accelerator, policy_model, f"checkpoints/policy_model_{global_step}")
                accelerator.wait_for_everyone()
                print(f"Model saved to checkpoints/policy_model_{global_step}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # Basic training arguments
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--num_gen", type=int, default=4, help="Number of generations per prompt")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--save_step", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    # Model and dataset configuration
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, default="You are a helpful math assistant.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    
    # Model initialization kwargs
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", 
                       choices=["float32", "float16", "bfloat16"], 
                       help="Model torch dtype")
    
    # vLLM configuration
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for generation")
    parser.add_argument("--vllm_server_url", type=str, help="URL for vLLM server")
    parser.add_argument("--ref_model_api_url", type=str, help="URL for reference model API")
    
    # Logging configuration
    parser.add_argument("--log_examples", action="store_true", 
                       help="Log generation examples to wandb")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process torch_dtype
    args.model_init_kwargs = {
        "torch_dtype": getattr(torch, args.torch_dtype),
        "use_cache": not args.gradient_checkpointing
    }
    
    main(args)