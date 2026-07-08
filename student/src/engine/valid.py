import torch
from tqdm import tqdm

from dataloader import cal_performance, prepare_batch_inputs, VARDataset
from utils import is_distributed, reduce_tensor, get_world_size, dist_log
import logging
logger = logging.getLogger(__name__)
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

    n_correct = pred.eq(gold).masked_select(valid_label_mask).sum().item()
    n_word = valid_label_mask.sum().item()
    return n_correct, n_word
# --- 修改结束(1) ---
'''
def eval_epoch(model, validation_data, device, opt):
    '''The same setting as training, where ground-truth word x_{t-1}
    is used to predict next word x_{t}, not realistic for real inference'''
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        dist_log('Start evaluating.......')

        for batch in validation_data:
            # * prepare data
            batched_data = prepare_batch_inputs(batch[0], device=device, non_blocking=opt.pin_memory)

            input_labels = batched_data['gt']
            # --- START OF CRITICAL FIX ---
            # 1. 准备所有输入，验证阶段不使用知识蒸馏，所以 teacher_logits=None
            forward_kwargs = {
                'encoder_input': batched_data['encoder_input'],
                'unmasked_encoder_input': batched_data['unmasked_encoder_input'],
                'input_ids_list': batched_data['decoder_input']['input_ids'],
                'input_masks_list': batched_data['decoder_input']['input_mask'],
                'token_type_ids_list': batched_data['decoder_input']['token_type_ids'],
                'input_labels_list': batched_data['gt'],
                'teacher_logits': None
            }

            loss, pred_scores_list = model(**forward_kwargs)

            #loss, pred_scores, gold_labels = model(**forward_kwargs)
            '''
            # --- [调试代码] 如果第一次遇到 batch，检查标签情况 ---
            if not debug_printed and batch_idx == 0:
                valid_count = (gold_labels != -1).sum().item()
                if opt.local_rank == 0:
                    print(f"\n[DEBUG VALID] Batch 0:")
                    print(f"  - Pred Shape: {pred_scores.shape}")
                    print(f"  - Gold Shape: {gold_labels.shape}")
                    print(f"  - Valid Labels Count: {valid_count}")
                    if valid_count == 0:
                        print(f"  - [WARNING] ALL LABELS ARE -1 IN VALIDATION BATCH!")
                debug_printed = True
            # --------------------------------------------------
            '''
            # * keep logs
            n_correct = 0
            n_word = 0
            for pred, gold in zip(pred_scores_list, input_labels[-1]):
                n_correct += cal_performance(pred, gold)
                valid_label_mask = gold.ne(VARDataset.IGNORE)
                n_word += valid_label_mask.sum()

            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss # 累加的是【未归一化】的巨大损失值

       
    if is_distributed() and get_world_size() > 1:
        reduced_n_word_total = reduce_tensor(n_word_total.float())
        reduced_total_loss = reduce_tensor(total_loss.float())
        reduced_n_word_correct = reduce_tensor(n_word_correct.float())
    else:
        reduced_n_word_total = n_word_total
        reduced_total_loss = total_loss
        reduced_n_word_correct = n_word_correct

    # [Debug Logic] 如果总单词数为0，说明所有标签都被mask了，或者没有验证数据
    if reduced_n_word_total == 0:
        if opt.local_rank == 0:
            logger.warning("[Validation Warning] Total valid words for evaluation is 0! Check your validation dataset labels.")
        return 0.0, 0.0
    # --- 核心修改点 (2): 恢复按单词数计算损失的逻辑 ---
    # 计算每个单词的平均损失 (loss per word)
    loss_per_word = 1.0 * reduced_total_loss / reduced_n_word_total if reduced_n_word_total > 0 else 0
    # 计算准确率
    accuracy = 1.0 * reduced_n_word_correct / reduced_n_word_total if reduced_n_word_total > 0 else 0

    
    # 返回归一化后的 loss_per_word
    return float(loss_per_word), float(accuracy)