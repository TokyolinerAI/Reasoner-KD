# D:/code/model/my/teacher/src/generate_soft_labels.py
import os
import sys
import argparse
import torch
import json
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.data.distributed import DistributedSampler  # 尽管此脚本通常单GPU运行，但保留导入以防万一

# 修正Python模块导入路径
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 假设此脚本在 src/ 目录下
_SRC_DIR = _CURRENT_DIR
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

# 导入您项目中的模块
from model import get_model
from dataloader.var_dataset import VARDataset, sentences_collate, prepare_batch_inputs
from utils.dist import dist_log, is_distributed, get_rank, barrier
from utils.misc import init_seed


def get_generation_args():
    parser = argparse.ArgumentParser(description="Generate K-layer soft labels in PKL format.")

    # 核心参数
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='Path to the trained teacher model checkpoint (.chkpt file).')
    parser.add_argument('--config_path', required=True, type=str,
                        help='Path to the model configuration file (.cfg.json) saved during training.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the output soft_labels.pkl and soft_labels_index.json file.')

    # 数据和模型加载参数
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing the splits data files (e.g., .../VAR/).')
    parser.add_argument('--glove_path', type=str,
                        help='Path to the GloVe embeddings directory.')
    parser.add_argument('--glove_version', type=str,
                        default='min3_vocab_glove_6B.pt',
                        help='Specific GloVe version file, e.g., min3_vocab_glove_6B.pt')
    parser.add_argument('--batch_size', type=int, default=16,  # 减小 batch_size 以防OOM
                        help='Batch size for inference. (Default: 16)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of worker processes for data loading. (Default: 2)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA.')

    return parser.parse_args()


def generate_soft_labels_pkl(opt):
    """Main function to generate and save K-layer soft labels."""
    device = torch.device("cuda" if not opt.no_cuda and torch.cuda.is_available() else "cpu")
    dist_log(f"Using device: {device}")

    # (设置DDP，尽管此脚本通常是单机)
    if is_distributed():
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        barrier()

    # 1. 从配置文件加载模型配置
    with open(opt.config_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    config_namespace = argparse.Namespace(**model_config)
    dist_log("Model configuration loaded successfully.")

    # 2. 加载训练数据集 (关键: 使用 'teacher' 模式)
    dist_log("Loading training dataset...")
    word2idx_path = None
    if opt.glove_path and opt.glove_version:
        word2idx_path = os.path.join(opt.glove_path,
                                     opt.glove_version.replace('vocab_glove', 'word2idx').replace('.pt', '.json'))

    train_dataset = VARDataset(
        dset_name=config_namespace.dset_name,
        data_dir=opt.data_dir,
        max_t_len=config_namespace.max_t_len,
        max_v_len=config_namespace.max_v_len,
        max_n_len=config_namespace.max_n_len,
        mode='train',
        K=config_namespace.K,
        word2idx_path=word2idx_path,
        model_mode='teacher'  # 确保不掩码
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=False) if is_distributed() else None

    train_loader = DataLoader(
        train_dataset,
        collate_fn=sentences_collate,
        batch_size=opt.batch_size,
        shuffle=False,  # 生成索引时必须为 False
        sampler=train_sampler,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    dist_log(f"Training dataset loaded with {len(train_dataset)} samples.")

    # 3. 初始化并加载教师模型
    config_namespace.vocab_size = len(train_dataset.word2idx)
    model = get_model(config_namespace)

    dist_log(f"Loading checkpoint from {opt.checkpoint_path}...")
    # 必须使用 weights_only=False 来加载包含 argparse.Namespace 等的旧 .chkpt 文件
    checkpoint = torch.load(opt.checkpoint_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint['model']
    # 处理 DDP 包装
    if all(key.startswith('module.') for key in state_dict):
        dist_log("Detected DataParallel (module.) prefix, removing it.")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model.to(device)
    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank],
                                                          find_unused_parameters=False)
    model.eval()
    dist_log(f"Teacher model loaded to device.")

    # 4. 遍历数据并生成软标签 (流式写入)
    if get_rank() <= 0:  # 只有主进程创建文件
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)

    output_pkl_path = os.path.join(opt.output_dir, 'soft_labels.pkl')
    output_index_path = os.path.join(opt.output_dir, 'soft_labels_index.json')

    index_map = {}

    # 仅主进程写入
    if get_rank() <= 0:
        # --- 核心修改点 3: 流式写入，解决CPU内存爆炸 ---
        with open(output_pkl_path, 'wb') as f_pkl, torch.no_grad():
            dist_log(f"Generating and streaming logits to {output_pkl_path}...")

            for batch_data, batch_meta in tqdm(train_loader, desc="Generating & Writing Soft Labels"):
                batched_data = prepare_batch_inputs(batch_data, device=device, non_blocking=False)

                B = batched_data['num_sen'].shape[0]
                S = config_namespace.max_n_len

                # --- 核心修改点 1: 接收 Logits 列表 ---
                model_to_run = model.module if is_distributed() else model
                # all_k_logits_gpu_list 现在是一个 *列表*, e.g.,
                # [ (B*S, 72, V), (B*S, 84, V), ... ]
                all_k_logits_gpu_list = model_to_run.forward(
                    encoder_input=batched_data['encoder_input'],
                    unmasked_encoder_input=batched_data['unmasked_encoder_input'],
                    input_ids_list=batched_data['decoder_input']['input_ids'],
                    input_masks_list=batched_data['decoder_input']['input_mask'],
                    token_type_ids_list=batched_data['decoder_input']['token_type_ids'],
                    input_labels_list=batched_data['gt'],
                    mode='generate_soft_labels'  # 关键模式
                )


                if all_k_logits_gpu_list is None:
                    dist_log(f"Warning: Model returned None. Skipping batch.")
                    continue

                # --- 核心修改点 2: 单独 Reshape 列表中的每个张量 ---
                K = len(all_k_logits_gpu_list)
                # logits_reshaped_gpu_list 包含:
                # [ (B, S, 72, V), (B, S, 84, V), ... ]
                logits_reshaped_gpu_list = []
                for k_tensor in all_k_logits_gpu_list:
                    # 获取当前张量的 L 和 V
                    L_k = k_tensor.shape[1]
                    V = k_tensor.shape[2]
                    # Reshape
                    reshaped_tensor = k_tensor.view(B, S, L_k, V)
                    logits_reshaped_gpu_list.append(reshaped_tensor)

                # --- 核心修改点 3 (续): 遍历批次，逐个样本写入磁盘 ---
                for i in range(B):  # 遍历批次中的每个样本
                    example_name = batch_meta[i][0]['name']
                    num_events = batched_data['num_sen'][i].item()  # 实际句子数量

                    # (1) 为该样本创建一个列表
                    sample_logits_list_cpu = []

                    # (2) 遍历 K 个解码器
                    for k_tensor_reshaped in logits_reshaped_gpu_list:
                        # (a) 在GPU上切片: (num_events, L_k, V)
                        sample_logits_k_gpu = k_tensor_reshaped[i, :num_events]

                        # (b) 移至CPU并转换为 float16
                        sample_logits_k_cpu = sample_logits_k_gpu.cpu().numpy().astype(np.float16)
                        sample_logits_list_cpu.append(sample_logits_k_cpu)

                    # --- 核心修改点 4: 序列化并保存 *列表* ---
                    current_offset = f_pkl.tell()

                    # (4) 序列化这个列表
                    serialized_data = pickle.dumps(sample_logits_list_cpu)
                    f_pkl.write(serialized_data)

                    # (5) 记录索引 (逻辑不变)
                    index_map[example_name] = {
                        'offset': current_offset,
                        'len': len(serialized_data)
                    }

        dist_log(f"Finished writing logits to {output_pkl_path}")

        # 5. 写入索引文件
        dist_log(f"Saving index map to {output_index_path}")
        with open(output_index_path, 'w') as f_json:
            json.dump(index_map, f_json)

    barrier()  # 确保所有进程都完成了
    dist_log("Generation of PKL soft labels complete.")


if __name__ == '__main__':
    args = get_generation_args()

    # 设置DDP环境变量 (如果需要)
    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # ...

    generate_soft_labels_pkl(args)