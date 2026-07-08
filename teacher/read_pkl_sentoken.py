import pickle
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from tqdm import tqdm

# --- 1. 配置区 ---

# 文件路径定义
PKL_FILE_PATH = "D:/code/model/my/teacher/soft_labels_pkl/soft_labels.pkl"
OUTPUT_TXT_PATH = "D:/code/model/my/teacher/soft_labels_pkl/soft_labels_sentence_analysis.txt"
WORD2IDX_PATH = "D:/code/model/my/data/VAR/vocab_feature/min3_word2idx_6B.json"

# --- 核心修改点 (1): 将模型超参数定义为变量 ---
# 这使得代码更具可读性，并且易于维护
MAX_V_LEN = 50
MAX_N_LEN = 12
MAX_T_LEN = 22
TEXT_START_INDEX = MAX_V_LEN + MAX_N_LEN  # 自动计算出 62

# 分析参数
TOP_K = 5

# --- 2. 初始化GPU设备和词汇表 (代码不变) ---

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("未检测到GPU，将使用CPU进行计算。")

try:
    with open(WORD2IDX_PATH, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    idx2word = {int(v): k for k, v in word2idx.items()}
    print(f"成功加载词汇表，共 {len(idx2word)} 个单词。")
except FileNotFoundError:
    print(f"错误: 找不到词汇表文件 '{WORD2IDX_PATH}'。")
    idx2word = {}

# --- 3. 流式读取、GPU处理并写入完整的句子分析 ---

sample_count = 0
try:
    with open(PKL_FILE_PATH, "rb") as pkl_file, \
            open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as txt_file:

        txt_file.write(f"对 {PKL_FILE_PATH} 的软标签进行句子生成部分分析\n")
        txt_file.write(f"显示文本生成部分 (共 {MAX_T_LEN} 个位置) 最可能的 Top-{TOP_K} 个单词及其概率\n")
        txt_file.write("=" * 80 + "\n\n")

        pbar = tqdm(desc="在GPU上分析样本", unit="sample")

        while True:
            try:
                sample_data_np = pickle.load(pkl_file)
                sample_data_gpu = torch.from_numpy(sample_data_np).to(device)

                probs_gpu = F.softmax(sample_data_gpu, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs_gpu, k=TOP_K, dim=-1)

                top_k_probs_cpu = top_k_probs.cpu().numpy()
                top_k_indices_cpu = top_k_indices.cpu().numpy()

                txt_file.write(f"--- 样本 {sample_count + 1} ---\n")
                txt_file.write(f"形状 (Events, Seq_Len, Vocab_Size): {sample_data_np.shape}\n\n")

                for event_idx in range(sample_data_np.shape[0]):
                    txt_file.write(f"  [事件 {event_idx + 1} 的句子预测]\n")

                    # --- 核心诊断代码 ---
                    # 这段代码只会在处理第一个样本的第一个事件时打印一次
                    if sample_count == 0 and event_idx == 0:
                        print("\n" + "=" * 50)
                        print(f"[内部诊断] 正在准备分析句子Token...")
                        print(f"[内部诊断] MAX_T_LEN 的值是: {MAX_T_LEN}")
                        print(f"[内部诊断] 内部循环将执行 range({MAX_T_LEN}) 次。")
                        print("=" * 50 + "\n")
                    # --- 诊断结束 ---
                    # --- 核心修改点 (2): 遍历完整的文本长度 (22个位置) ---
                    for word_pos in range(MAX_T_LEN):
                        # --- 核心修改点 (3): 使用计算好的变量进行索引 ---
                        # current_pos = 62, 63, ..., 83
                        current_pos = TEXT_START_INDEX + word_pos

                        indices = top_k_indices_cpu[event_idx][current_pos]
                        probabilities = top_k_probs_cpu[event_idx][current_pos]

                        predictions = []
                        for idx, prob in zip(indices, probabilities):
                            word = idx2word.get(idx, f"[未知ID:{idx}]")
                            predictions.append(f"'{word}' ({prob:.2%})")

                        # --- 核心修改点 (4): 使用更直观的 "Word" 标签 ---
                        txt_file.write(f"    Word {word_pos + 1}:  {', '.join(predictions)}\n")
                    txt_file.write("\n")

                txt_file.write("=" * 80 + "\n\n")

                sample_count += 1
                pbar.update(1)

            except EOFError:
                break

    pbar.close()
    print("\n分析完成！")
    print(f"总共分析了 {sample_count} 个样本。")
    print(f"分析结果已保存到: {OUTPUT_TXT_PATH}")

except FileNotFoundError:
    print(f"错误: 找不到PKL文件 '{PKL_FILE_PATH}'")
except Exception as e:
    print(f"处理过程中发生未知错误: {e}")