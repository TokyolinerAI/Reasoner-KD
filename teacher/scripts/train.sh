#!/usr/bin/env bash

# 检查是否提供了必要的参数
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "错误: 请提供 GPU 数量 和 model_name."
  echo "用法: bash teacher/scripts/train.sh <N_GPUS> <model_name> [其他参数...]"
  exit 1
fi

# --- 参数设置 ---
GPUS=$1
MODEL_NAME=$2
MASTER_PORT=29500

echo "---------------------------------------------------------"
echo ">>>>>>>> 在 VAR 数据集上运行训练"
echo "使用 GPU 数量: ${GPUS}"
echo "模型名称: ${MODEL_NAME}"
echo "---------------------------------------------------------"

# --- 路径和超参数设置 ---
MAX_N_LEN=12
MAX_T_LEN=22
MAX_V_LEN=50


PROJECT_ROOT=$(dirname "$0")/..
DATA_DIR="${PROJECT_ROOT}/../data/VAR"
GLOVE_PATH="${DATA_DIR}/vocab_feature/"
GLOVE_VERSION="min3_vocab_glove_6B.pt"

# --- 核心：确保调用的是顶层的 run.py ---
RUN_SCRIPT="${PROJECT_ROOT}/run.py"

# --- 核心修改：根据GPU数量选择启动方式 ---

# 准备通用参数
COMMON_ARGS="
    --dset_name VAR \
    --model_name ${MODEL_NAME} \
    --data_dir ${DATA_DIR} \
    --glove_path ${GLOVE_PATH} \
    --glove_version ${GLOVE_VERSION} \
    --max_n_len ${MAX_N_LEN} \
    --max_t_len ${MAX_T_LEN} \
    --max_v_len ${MAX_V_LEN} \
    ${@:3}
"

if [ "${GPUS}" -gt 1 ]; then
  # --- 多GPU模式：使用 torchrun ---
  echo "检测到多个GPU，使用分布式模式 (torchrun) 启动..."
  export USE_LIBUV=0
  torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS} --master_port=${MASTER_PORT} \
      ${RUN_SCRIPT} ${COMMON_ARGS} # <-- 修改这里
else
  # --- 单GPU模式：直接使用 python ---
  echo "检测到单个GPU，使用标准模式 (python) 启动..."
  python ${RUN_SCRIPT} ${COMMON_ARGS} # <-- 修改这里
fi