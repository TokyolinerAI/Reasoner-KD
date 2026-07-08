import pickle
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from tqdm import tqdm  # 引入tqdm来显示进度条

# --- 1. 配置区 ---

# 文件路径定义
PKL_FILE_PATH = "D:/code/model/my/teacher/soft_labels_pkl/soft_labels.pkl"
OUTPUT_TXT_PATH = "D:/code/model/my/teacher/soft_labels_pkl/analysis_22word_a_sentence.txt"
# !!重要!!: 需要word2idx文件来将预测的ID转换为可读的单词
WORD2IDX_PATH = "D:/code/model/my/data/VAR/vocab_feature/min3_word2idx_6B.json"

# 分析参数
TOP_K = 5  # 我们希望查看每个位置最可能的 K 个单词

# --- 2. 初始化GPU设备和词汇表 ---

# 检查是否有可用的GPU，否则回退到CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("未检测到GPU，将使用CPU进行计算。")

# 加载词汇表以进行ID到单词的转换
try:
    with open(WORD2IDX_PATH, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    # 创建一个反向映射，用于从索引查找单词
    idx2word = {int(v): k for k, v in word2idx.items()}
    print(f"成功加载词汇表，共 {len(idx2word)} 个单词。")
except FileNotFoundError:
    print(f"错误: 找不到词汇表文件 '{WORD2IDX_PATH}'。无法将ID转换为单词。")
    # 创建一个空的idx2word，程序可以继续运行但无法显示单词
    idx2word = {}

# --- 3. 流式读取、GPU处理并写入分析结果 ---

sample_count = 0
try:
    # 同时以二进制读模式打开PKL文件，以写模式打开TXT输出文件
    with open(PKL_FILE_PATH, "rb") as pkl_file, \
            open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as txt_file:

        txt_file.write(f"对 {PKL_FILE_PATH} 的软标签进行GPU分析\n")
        txt_file.write(f"显示每个位置最可能的 Top-{TOP_K} 个单词及其概率\n")
        txt_file.write("=" * 80 + "\n\n")

        # 使用tqdm包装循环，以提供可视化的进度条
        pbar = tqdm(desc="在GPU上分析样本", unit="sample")

        # 核心逻辑：在一个无限循环中处理文件流
        while True:
            try:
                # a. 从文件中加载 *一个* 样本到CPU内存 (这是一个Numpy数组)
                sample_data_np = pickle.load(pkl_file)

                # b. 将Numpy数组转换为PyTorch张量，并立即发送到GPU
                sample_data_gpu = torch.from_numpy(sample_data_np).to(device)

                # c. 在GPU上执行计算
                #    - 应用Softmax将Logits转换为概率
                #    - 使用torch.topk高效找到最可能的K个结果
                probs_gpu = F.softmax(sample_data_gpu, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs_gpu, k=TOP_K, dim=-1)

                # d. 将 *小规模的计算结果* 传回CPU，以便写入文件
                top_k_probs_cpu = top_k_probs.cpu().numpy()
                top_k_indices_cpu = top_k_indices.cpu().numpy()

                # e. 格式化并写入分析结果到TXT文件
                txt_file.write(f"--- 样本 {sample_count + 1} ---\n")
                txt_file.write(f"形状 (Events, Seq_Len, Vocab_Size): {sample_data_np.shape}\n\n")

                # 遍历样本中的每个事件（句子）
                for event_idx in range(sample_data_np.shape[0]):
                    txt_file.write(f"  [事件 {event_idx + 1}]\n")
                    num_tokens = sample_data_np.shape[1]

                    # 为了避免文件过大，只显示前10个token位置的分析
                    for token_pos in range(min(10, num_tokens)):
                        # 获取当前位置的top-k索引和概率
                        indices = top_k_indices_cpu[event_idx][token_pos+62]
                        probabilities = top_k_probs_cpu[event_idx][token_pos+62]

                        # 构建可读的字符串
                        predictions = []
                        for idx, prob in zip(indices, probabilities):
                            word = idx2word.get(idx, f"[未知ID:{idx}]")
                            predictions.append(f"'{word}' ({prob:.2%})")

                        txt_file.write(f"    Token {token_pos}:  {', '.join(predictions)}\n")
                    txt_file.write("\n")

                txt_file.write("=" * 80 + "\n\n")

                sample_count += 1
                pbar.update(1)  # 更新进度条

            except EOFError:
                # 文件读取到末尾，正常退出循环
                break

    pbar.close()
    print("\n分析完成！")
    print(f"总共分析了 {sample_count} 个样本。")
    print(f"分析结果已保存到: {OUTPUT_TXT_PATH}")

except FileNotFoundError:
    print(f"错误: 找不到PKL文件 '{PKL_FILE_PATH}'")
except Exception as e:
    print(f"处理过程中发生未知错误: {e}")