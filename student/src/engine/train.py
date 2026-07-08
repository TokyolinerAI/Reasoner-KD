import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader.var_dataset import cal_performance, VARDataset, prepare_batch_inputs
from utils import is_distributed, reduce_tensor,get_world_size, dist_log
'''
def cal_performance(pred, gold):
    """
    计算性能指标的辅助函数，返回正确的单词数和总单词数
    pred: (N, L, V)
    gold: (N, L)
    """
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    valid_label_mask = gold.ne(VARDataset.IGNORE)

    n_correct = pred.eq(gold).masked_select(valid_label_mask).sum()
    n_word = valid_label_mask.sum()

    return n_correct, n_word
'''
def train_epoch(model, training_data, optimizer, ema, device, opt, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    # 获取实际的模型实例 (处理分布式训练的封装)
    actual_model = model.module if hasattr(model, 'module') else model

    if not opt.disable_scheduled_sampling:
        actual_model.set_epoch(epoch)

    # 在分布式训练时，只在主进程（rank 0）显示tqdm进度条
    data_iterator = tqdm(training_data, desc=f"Training Epoch {epoch}", disable=(opt.local_rank != 0))

    for batch_idx, batch in enumerate(data_iterator):
        niter = epoch * len(training_data) + batch_idx

        batched_data = prepare_batch_inputs(batch[0], device=device, non_blocking=opt.pin_memory)

        input_labels = batched_data['gt']

        optimizer.zero_grad()

        # --- START OF CRITICAL FIX ---
        # 1. 一次性构建好所有需要传递给模型的参数
        forward_kwargs = {
            'encoder_input': batched_data['encoder_input'],
            'unmasked_encoder_input': batched_data['unmasked_encoder_input'],
            'input_ids_list': batched_data['decoder_input']['input_ids'],
            'input_masks_list': batched_data['decoder_input']['input_mask'],
            'token_type_ids_list': batched_data['decoder_input']['token_type_ids'],
            'input_labels_list': batched_data['gt'],
            'teacher_logits': batched_data['teacher_logits']
        }
        '''
        # 2. 如果启用蒸馏，则从数据加载器中添加教师logits
        if opt.use_distillation:
            if 'teacher_logits' in batched_data:
                forward_kwargs['teacher_logits'] = batched_data['teacher_logits']
            else:
                raise ValueError("Knowledge distillation is enabled, but 'teacher_logits' not found in the batch.")
        else:
            # 如果不蒸馏，确保teacher_logits为None
            forward_kwargs['teacher_logits'] = None
        '''
        # 3. 只有一个统一的模型调用，模型现在只返回总损失
        loss, pred_scores_list = model(**forward_kwargs)
        # --- END OF CRITICAL FIX ---
        # --- 核心修改点：恢复归一化步骤 ---
        # 与原始 teacher/src/engine/train.py 的逻辑保持一致
        if loss is not None and not torch.isnan(loss):
            # 获取批次中的总句子数
            num_sen = batched_data['num_sen'].detach().sum()

            # 处理分布式训练的情况
            if is_distributed() and get_world_size() > 1:
                reduced_n_sen = reduce_tensor(num_sen.float())
            else:
                reduced_n_sen = num_sen

            # 第一次归一化：按句子数和GPU数量
            # 加上一个很小的数避免除以零
            loss = loss / (reduced_n_sen.float().item() * get_world_size() + 1e-8)

            # 第二次归一化：按批次大小
            # (注意: 原始代码的第二次归一化可能值得商榷，因为它可能导致过度归一化。
            # 我们可以先保留它以对齐原始行为，如果ppl过低再考虑移除)
            loss = loss / float(opt.batch_size)
        # --- 修改结束 ---

        if not torch.isnan(loss):
            loss.backward()

        if opt.grad_clip != -1:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

        optimizer.step()

        # 更新EMA模型
        if ema is not None:
            ema(model, niter)

        # * keep logs
        n_correct = 0
        n_word = 0
        for pred, gold in zip(pred_scores_list, input_labels[-1]):
            n_correct += cal_performance(pred, gold)
            valid_label_mask = gold.ne(VARDataset.IGNORE)
            n_word += valid_label_mask.sum()

        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss

        if opt.local_rank == 0:
            data_iterator.set_postfix(
                loss=loss.item(),
                acc=f"{(n_correct / n_word if n_word > 0 else 0):.4f}"
            )

    if is_distributed() and get_world_size() > 1:
        reduced_n_word_total = reduce_tensor(n_word_total.float())
        reduced_total_loss = reduce_tensor(total_loss.float())
        reduced_n_word_correct = reduce_tensor(n_word_correct.float())
    else:
        reduced_total_loss = total_loss
        reduced_n_word_total = n_word_total
        reduced_n_word_correct = n_word_correct

    # 计算每个单词的平均损失 (loss per word)
    loss_per_word = 1.0 * reduced_total_loss / reduced_n_word_total if reduced_n_word_total > 0 else 0
    # 计算准确率
    accuracy = 1.0 * reduced_n_word_correct / reduced_n_word_total if reduced_n_word_total > 0 else 0

    # 返回归一化后的 loss_per_word
    return float(loss_per_word), float(accuracy)