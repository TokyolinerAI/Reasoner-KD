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
OUTPUT_TXT_PATH = "D:/code/model/my/teacher/soft_labels_pkl/analysis_all_decoders.txt"
# !!重要!!: 需要word2idx文件来将预测的ID转换为可读的单词
WORD2IDX_PATH = "D:/code/model/my/data/VAR/vocab_feature/min3_word2idx_6B.json"

# 分析参数
TOP_K = 5  # 我们希望查看每个位置最可能的 K 个单词

# !!重要!!: 这些值必须与训练教师模型时的设置一致
# 它们决定了我们如何找到文本部分的开始位置
MAX_V_LEN = 50  # 假设: 视频标记长度
MAX_N_LEN = 12  # 假设: 句子标记长度 (仅 K > 0 时使用)

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
                # --- 核心修正 (1/4) ---
                # a. 加载 *一个* 样本数据。
                # 这现在是一个 *列表*，例如 [ array_K0, array_K1, ... ]
                sample_data_list_np = pickle.load(pkl_file)

                K = len(sample_data_list_np)  # 解码器数量
                num_events = sample_data_list_np[0].shape[0]  # 事件数量

                txt_file.write(f"--- 样本 {sample_count + 1} (共 {num_events} 个事件, {K} 个解码器) ---\n")

                # --- 核心修正 (2/4) ---
                # b. 遍历列表中的 *每个* 解码器
                for k_idx, decoder_logits_np in enumerate(sample_data_list_np):

                    # (形状: NumEvents, L_k, Vocab_Size)
                    (Ev, L_k, V) = decoder_logits_np.shape
                    txt_file.write(f"\n  [解碼器 {k_idx}] (Logits 形状: {Ev}, {L_k}, {V})\n")

                    # c. 将 *这一个* 解码器的Numpy数组发送到GPU
                    # (使用 .to(torch.float32) 是因为 float16 无法进行 softmax)
                    decoder_logits_gpu = torch.from_numpy(decoder_logits_np).to(torch.float32).to(device)

                    # d. 在GPU上执行计算
                    probs_gpu = F.softmax(decoder_logits_gpu, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs_gpu, k=TOP_K, dim=-1)

                    # e. 将 *小规模的计算结果* 传回CPU
                    top_k_probs_cpu = top_k_probs.cpu().numpy()
                    top_k_indices_cpu = top_k_indices.cpu().numpy()

                    # --- 核心修正 (3/4) ---
                    # f. 智能确定分析位置，而不是硬编码的 +62
                    #    我们分析第一个 "真实" 文本标记 [BOS] 的位置

                    if k_idx == 0:
                        # 解码器 0: [VID_tokens] + [BOS] + [TEXT_tokens]...
                        # L_k 应该是 72 (50+22)
                        bos_token_pos = MAX_V_LEN + 1
                    else:
                        # 解码器 1+: [VID_tokens] + [SEN_tokens] + [BOS] + [TEXT_tokens]...
                        # L_k 应该是 84 (50+12+22)
                        bos_token_pos = MAX_V_LEN + MAX_N_LEN + 1

                    if bos_token_pos >= L_k:
                        txt_file.write(f"    错误: 计算的 [BOS] 位置 ({bos_token_pos}) 超出了序列长度 ({L_k})!\n")
                        continue

                    # g. 遍历样本中的每个事件（句子）
                    # 为了避免文件过大，我们只分析 *第一个事件 (event_idx=0)*
                    event_idx = 0
                    txt_file.write(f"    分析 [事件 {event_idx + 1}] 在 [BOS] (位置 {bos_token_pos}) 处的预测:\n")

                    # --- 核心修正 (4/4) ---
                    #    获取 *该事件* 和 *该位置* 的 top-k 索引和概率
                    indices = top_k_indices_cpu[event_idx][bos_token_pos]
                    probabilities = top_k_probs_cpu[event_idx][bos_token_pos]

                    # 构建可读的字符串
                    predictions = []
                    for idx, prob in zip(indices, probabilities):
                        word = idx2word.get(idx, f"[未知ID:{idx}]")
                        predictions.append(f"'{word}' ({prob:.2%})")

                    txt_file.write(f"      Token @{bos_token_pos}:  {', '.join(predictions)}\n")

                txt_file.write("\n" + "=" * 80 + "\n\n")

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
    print(f"错误: 找不到PKL文件 '{PKL_FILE_PATH}' 或 '{WORD2IDX_PATH}'")
except Exception as e:
    print(f"处理过程中发生未知错误: {e}")