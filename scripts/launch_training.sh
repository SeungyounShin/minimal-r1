#!/bin/bash

# 获取当前主机名
HOSTNAME=$(hostname)

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 获取项目根目录
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 配置文件路径
CONFIG_FILE="$PROJECT_ROOT/config/nodes.yaml"
ACCELERATE_CONFIG_FILE="$PROJECT_ROOT/config/accelerate_config.yaml"

# 解析YAML配置文件的函数
parse_yaml() {
    local prefix=$2
    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
    sed -ne "s|^\($s\):|\1|" \
         -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
         -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
    awk -F$fs '{
        indent = length($1)/2;
        vname[indent] = $2;
        for (i in vname) {if (i > indent) {delete vname[i]}}
        if (length($3) > 0) {
            vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
            printf("%s%s%s=\"%s\"\n", "'$prefix'",vn,$2,$3);
        }
    }'
}

# 加载配置
eval $(parse_yaml "$CONFIG_FILE")

# 设置环境变量
export PDSH_RCMD_TYPE=ssh
export MASTER_ADDR=$training_master_node
export MASTER_PORT=$training_master_port

# 启动训练
echo "Launching training on $HOSTNAME..."

# 使用accelerate启动训练
accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $training_master_node \
    --main_process_port $training_master_port \
    src/train_grpo.py \
    --config $CONFIG_FILE \
    --dataset_name "AI-MO/NuminaMath-TIR" \
    --epochs 3 \
    --batch_size 4 \
    --lr 1e-5 \
    --beta 0.1 \
    --temperature 0.7 \
    --num_gen 4 \
    --max_tokens 512 \
    --torch_dtype bfloat16 \
    --use_vllm \
    --vllm_server_url "${services_generation_url}" \
    --ref_model_api_url "${services_reference_url}" \
    --gradient_checkpointing \
    --log_examples
