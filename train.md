# xDAN-RLAIF-GRPO 训练指南

## 1. 配置文件

### 1.1 节点配置 (`config/nodes.yaml`)
```yaml
# 训练节点配置
training:
  master_node: "gpu007"
  master_port: 29500
  # 远程服务配置
  services:
    generation:
      url: "http://gpu004:8000"  # vLLM生成服务URL
    reference:
      url: "http://gpu008:8001"  # 参考模型服务URL

# 模型配置
models:
  base_path: "/data/vayu/train/models"  # 模型基础路径
  train:
    name: "xDAN-L1-Qwen25-7B-Instruct"  # 训练使用的模型名称
    path: "/data/vayu/train/models/xDAN-L1-Qwen25-7B-Instruct"  # 完整路径
  output:
    path: "output"  # 相对于项目根目录的输出路径
    checkpoint_dir: "checkpoints"  # checkpoint保存目录
    log_dir: "logs"  # 日志目录
```

### 1.2 训练节点配置 (`config/hostfile`)
```text
# 训练节点配置
# 格式: hostname slots=num_gpus
gpu007 slots=8  # 主节点
gpu005 slots=8  # worker节点
```

## 2. 启动服务

### 2.1 启动vLLM生成服务 (在gpu004上)
```bash
# 使用8张GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/launch_vllm.py \
    --model_path /data/vayu/train/models/xDAN-L1-Qwen25-7B-Instruct \
    --port 8000 \
    --tensor_parallel_size 8
```

### 2.2 启动Reference服务 (在gpu008上)
```bash
# 使用8张GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/launch_ref_model.py \
    --model_path /data/vayu/train/models/xDAN-L1-Qwen25-7B-Instruct \
    --port 8001 \
    --tensor_parallel_size 8
```

## 3. 启动训练

### 3.1 启动分布式训练
```bash
# 在任意节点执行
./scripts/launch_distributed.sh
```

这个命令会：
1. 读取`config/nodes.yaml`中的配置
2. 在主节点(gpu007)上启动训练
3. 通过DeepSpeed自动管理worker节点(gpu005)

## 4. 目录结构
```
项目根目录/
├── config/
│   ├── nodes.yaml      # 节点和模型配置
│   ├── hostfile        # DeepSpeed训练节点配置
│   ├── accelerate_config.yaml  # Accelerate配置
│   └── deepspeed/
│       └── zero3.json  # DeepSpeed ZeRO-3配置
├── src/
│   ├── train_grpo.py   # 训练主脚本
│   ├── launch_vllm.py  # vLLM服务启动脚本
│   └── launch_ref_model.py  # 参考模型服务启动脚本
├── scripts/
│   ├── launch_distributed.sh  # 分布式训练启动脚本
│   └── launch_training.sh     # 训练启动脚本
└── output/             # 训练输出目录
    ├── checkpoints/   # 模型检查点
    └── logs/         # 训练日志
```

## 5. 验证服务状态

### 5.1 验证vLLM服务
```bash
curl http://gpu004:8000/health
# 应返回: {"status": "healthy"}
```

### 5.2 验证Reference服务
```bash
curl http://gpu008:8001/health
# 应返回: {"status": "healthy"}
```

### 5.3 验证训练进程
```bash
# 在gpu007上执行
ps aux | grep train_grpo.py
```

## 6. 注意事项

1. 确保所有节点都能相互访问
2. 确保模型路径在所有节点上都存在
3. 确保所有节点的CUDA环境正确配置
4. 训练日志会保存在`output/logs/`目录
5. 模型检查点会保存在`output/checkpoints/`目录
6. 如果需要修改训练配置，编辑`config/nodes.yaml`
7. 如果需要修改节点配置，编辑`config/hostfile`