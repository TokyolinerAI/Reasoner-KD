import sys
import json
import copy
import math
import token

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from easydict import EasyDict as edict
from .utils.optimization import LabelSmoothingLoss
from .modules import BertLayerNorm, BertLMPredictionHead
from .visual_model import VideoEncodingTrans
from .linguistic_model import TextEmbedding, BertEmbeddingsWithVideo, BertEmbeddingsWithSentence, BertEncoder, \
    BertEncoderSen
from utils import dist_log

import logging
logger = logging.getLogger(__name__)

"""
a) 目标类别知识蒸馏 (TCKD / SAKD): 单独蒸馏正确答案的概率。论文中提到使用“二元分类损失”。这强制学生模型学习教师对正确动作的判断有多么自信。这关注的是“局部精准度”。
b) 非目标类别知识蒸馏 (NCKD / SDKD): 单独蒸馏所有错误答案的概率分布。论文中提到使用“MSE损失”。这强制学生模型学习教师认为“哪些错误与正确答案更相似”，从而理解动作间的全局关系。这关注的是“全局规划能力”。
最终损失: L_total = α * TCKD + β * NCKD，分别对这两种知识进行加权。
"""
# --- START OF MODIFICATION (1/2): 实现论文中的解耦知识蒸馏损失 ---
class DecoupledKnowledgeDistillationLoss(nn.Module):
    """
    Implements Decoupled Knowledge Distillation (PKDD) from the AAAI-25 paper.
    L_iikd = alpha * L_sa + beta * L_sd
    """

    # 修正: 移除默认值，强制必须传入，以便调试
    def __init__(self, vocab_size, ignore_index=-1,
                 alpha=None, beta=None, temperature=None):
        super(DecoupledKnowledgeDistillationLoss, self).__init__()
        # 增加断言，确保参数被正确传递
        assert alpha is not None, "SAKD weight (alpha) must be provided."
        assert beta is not None, "SDKD weight (beta) must be provided."
        assert temperature is not None, "Temperature must be provided."

        self.alpha = alpha
        self.beta = beta
        self.T = temperature
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size

        # SAKD: Single Action Knowledge Distillation Loss (使用BCE)
        self.sakd_loss_fn = nn.BCELoss(reduction='sum')

        # SDKD: Sequence Distribution Knowledge Distillation Loss (使用MSE)
        self.sdkd_loss_fn = nn.MSELoss(reduction='sum')

    def forward(self, student_logits, hard_labels, teacher_logits):
        # 1. 维度对齐检查 (展平前)
        # student_logits: (N, L, V) 或 (N, V)
        # teacher_logits: (N, L, V) 或 (N, V)

        # 统一展平为 (Total_Tokens, Vocab_Size)
        S_V = student_logits.reshape(-1, self.vocab_size)
        T_V = teacher_logits.reshape(-1, self.vocab_size)
        H_L = hard_labels.reshape(-1)

        # 再次兜底检查长度 (防止对齐函数未覆盖的情况)
        if T_V.size(0) != H_L.size(0):
             min_len = min(T_V.size(0), H_L.size(0))
             S_V = S_V[:min_len]
             T_V = T_V[:min_len]
             H_L = H_L[:min_len]
        # 1. 创建有效位置的掩码 (忽略padding)
        valid_mask = (H_L != self.ignore_index).view(-1)
        if valid_mask.sum() == 0:
            zero_loss = torch.tensor(0.0, device=student_logits.device)
            return zero_loss, zero_loss, zero_loss

        # 3. 提取有效数据
        valid_student_logits = S_V[valid_mask]
        valid_teacher_logits = T_V[valid_mask]
        valid_hard_labels = H_L[valid_mask]

        # --- a) SAKD: 目标类别知识蒸馏 ---
        # 获取每个样本正确类别的one-hot编码
        gt_mask = F.one_hot(valid_hard_labels, num_classes=self.vocab_size).float()

        # 预计算 Softmax 概率 (PKDD 需要先计算全局概率)
        pred_student = F.softmax(valid_student_logits / self.T, dim=1)
        pred_teacher = F.softmax(valid_teacher_logits / self.T, dim=1)

        # --- a) SAKD (TCKD): 目标类别知识蒸馏 ---
        # PKDD 原理: 将分布视为二元分类问题 (目标类 vs 非目标类)
        # 计算目标类的概率和
        p_s_target = (pred_student * gt_mask).sum(dim=1, keepdim=True)
        p_t_target = (pred_teacher * gt_mask).sum(dim=1, keepdim=True)

        # [核心修改 2] 使用 BCE 计算二元分布差异
        # 注意：BCELoss 会自动计算 -(y*log(x) + (1-y)*log(1-x))，
        # 所以我们只需要传入目标类的概率即可，它等价于计算了 [p, 1-p] 的分布差异。
        # 为了数值稳定性，限制范围防止 log(0)
        p_s_target = torch.clamp(p_s_target, min=1e-7, max=1.0 - 1e-7)
        p_t_target = torch.clamp(p_t_target, min=1e-7, max=1.0 - 1e-7)
        
        loss_sa = self.sakd_loss_fn(p_s_target, p_t_target)
        loss_sa = loss_sa * (self.T ** 2)

        # --- b) SDKD: 非目标类别知识蒸馏 ---
        # 关键: 使用masked_fill屏蔽目标类别，而不是masked_select
        # PKDD 原理: 排除目标类，对剩余类别的分布进行 MSE 匹配

        masked_student_logits = valid_student_logits.masked_fill(gt_mask.bool(), -float('inf'))
        masked_teacher_logits = valid_teacher_logits.masked_fill(gt_mask.bool(), -float('inf'))

        # [核心修改 3] 重新计算 Softmax (仅在非目标类上归一化)
        pred_student_nckd = F.softmax(masked_student_logits / self.T, dim=1)
        pred_teacher_nckd = F.softmax(masked_teacher_logits / self.T, dim=1)

        # [核心修改 4] 使用 MSE 损失
        loss_sd = self.sdkd_loss_fn(pred_student_nckd, pred_teacher_nckd)
        loss_sd = loss_sd * (self.T ** 2)

        #loss_sd = loss_sd * self.vocab_size
        # --- 组合最终的解耦蒸馏损失 (对应论文公式7 L_iikd) ---
        total_distillation_loss = self.alpha * loss_sa + self.beta * loss_sd

        # --- START OF MODIFICATION (1/2): 返回所有损失分量 ---
        return total_distillation_loss, loss_sa.detach(), loss_sd.detach()
        # --- END OF MODIFICATION (1/2) ---


# --- END OF MODIFICATION (1/2) ---

def _concat_list_of_tensors(_list):
    outs = []
    for param in _list:
        out = torch.stack(param, dim=0).contiguous().view(-1, *param[0].shape[1:])
        outs.append(out)
    return outs


class Reasoner(nn.Module):
    def __init__(self, config):
        super(Reasoner, self).__init__()
        self.config = config

        self.video_encoder = VideoEncodingTrans(config, add_postion_embeddings=True)
        self.projection_head_q = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(),
                                               nn.Linear(config.hidden_size, config.hidden_size))

        self.video_encoder_aux = VideoEncodingTrans(config, add_postion_embeddings=True)
        self.projection_head_k = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(),
                                               nn.Linear(config.hidden_size, config.hidden_size))
        self.m = self.config.momentum_aux_m

        for q, k in zip((self.video_encoder.parameters(), self.projection_head_q.parameters()),
                        (self.video_encoder_aux.parameters(), self.projection_head_k.parameters())):
            for param_q, param_k in zip(q, k):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        if self.config.K > 1:
            shared_embeddings_dec = BertEmbeddingsWithSentence(config, add_postion_embeddings=True)
            shared_decoder = BertEncoderSen(config)
            shared_pred_head = BertLMPredictionHead(config, None)
            self.embeddings_decs = nn.ModuleList([
                BertEmbeddingsWithVideo(config, add_postion_embeddings=True) if k == 0 else shared_embeddings_dec
                for k in range(self.config.K)])
            self.decoders = nn.ModuleList(
                [BertEncoder(config) if k == 0 else shared_decoder for k in range(self.config.K)])
            self.pred_heads = nn.ModuleList(
                [BertLMPredictionHead(config, None) if k == 0 else shared_pred_head for k in range(self.config.K)])
        else:
            self.embeddings_decs = nn.ModuleList([BertEmbeddingsWithVideo(config, add_postion_embeddings=True)])
            self.decoders = nn.ModuleList([BertEncoder(config)])
            self.pred_heads = nn.ModuleList([BertLMPredictionHead(config, None)])

        # --- START OF MODIFICATION (2/2): 初始化新的损失函数 ---
        # 硬损失函数 (L_hard)
        self.hard_loss_fn = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=-1) \
            if "label_smoothing" in config and config.label_smoothing > 0 else nn.CrossEntropyLoss(ignore_index=-1)
        
        #初始化内部步数计数器
        self.current_step_counter = 0

        # 如果使用蒸馏，初始化解耦蒸馏损失函数
        if self.config.use_distillation:
            logger.info("Using Decoupled Knowledge Distillation Loss (SAKD + SDKD).")
            # 诊断代码: 打印接收到的参数值
            
            print("=" * 50)
            print(f"[DIAGNOSIS] Initializing Distillation Loss:")
            print(f"[DIAGNOSIS]   alpha_sa_loss from config = {self.config.alpha_sa_loss}")
            print(f"[DIAGNOSIS]   beta_sd_loss from config  = {self.config.beta_sd_loss}")
            print("=" * 50)
            
            self.distill_loss_fn = DecoupledKnowledgeDistillationLoss(
                self.config.vocab_size,
                ignore_index=-1,
                alpha=self.config.alpha_sa_loss,
                beta=self.config.beta_sd_loss,
                temperature=self.config.temperature_kd
            )
        # --- END OF MODIFICATION (2/2) ---

        self.apply(self.init_bert_weights)
        self.probs = 0.
        self.conf_bucket_size = config.conf_bucket_size
        if self.config.sentence_emb_aggregation_mode == 'weighted':
            self._weighted_para = nn.Sequential(nn.Linear(config.hidden_size, 4 * config.hidden_size), nn.GELU(),
                                                nn.Linear(4 * config.hidden_size, 1))

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

        # * scheduled sampling

    def _update_scheduled_sampling(self, _pred, input_ids):
        _pred = torch.cat(
            [(torch.zeros((int(_pred.shape[0]), 1, int(_pred.shape[2])))).to(_pred.device).float(), _pred], dim=1)[:,
                :-1]
        # * pred_size S*B,L,vocab_size -> S*B,L
        pred = _pred.max(2)[1]
        # * for each word, p prob to be replaced by predicted results
        # * S*B,L
        prob = torch.softmax(_pred, dim=2).max(2)[0]
        replace_mask = (prob > self.config.conf_replace_tr)
        # * DO NOT replace video ids + [BOS]
        replace_mask[:, :(self.config.max_v_len + 1)] = False

        input_ids[replace_mask] = pred[replace_mask]

        return input_ids

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.decay_prob(epoch, k=self.config.decay_k)

        #每个 Epoch 开始时重置计数器
        self.current_step_counter = 0

    def decay_prob(self, i, or_type=3, k=3000):
        if or_type == 1:  # Linear decay
            or_prob_begin, or_prob_end = 1., 0.
            or_decay_rate = (or_prob_begin - or_prob_end) / 10.
            ss_decay_rate = 0.1
            prob = or_prob_begin - (ss_decay_rate * i)
            if prob < or_prob_end:
                prob_i = or_prob_end
                dist_log('[Linear] schedule sampling probability do not change {}'.format(prob_i))
            else:
                prob_i = prob
                dist_log('[Linear] decay schedule sampling probability to {}'.format(prob_i))

        elif or_type == 2:  # Exponential decay
            prob_i = np.power(k, i)
            dist_log('[Exponential] decay schedule sampling probability to {}'.format(prob_i))

        elif or_type == 3:  # Inverse sigmoid decay
            prob_i = k / (k + np.exp((i / k)))
            dist_log('[Inverse] decay schedule sampling probability to {}'.format(prob_i))
        self.probs = prob_i

        return prob_i

    def get_word_orcale_tokens(self, _pred, prev_output_tokens, epsilon=1e-6):
        _gumbel_noise = 0.5

        B, L = prev_output_tokens.size()
        pred_logits = _pred[:, self.config.max_v_len:]
        # B x L x V
        pred_logits.add_(-torch.log(-torch.log(torch.Tensor(
            pred_logits.size()).to(pred_logits.device).uniform_(0, 1) + epsilon) + epsilon)) / _gumbel_noise

        pred_tokens = torch.max(pred_logits, dim=-1)[1]
        bos_idx = prev_output_tokens[0, self.config.max_v_len]
        pred_tokens = torch.cat([(bos_idx * torch.ones((B, 1)).to(pred_tokens.device)), pred_tokens], dim=1)[:, :-1]

        sample_gold_prob = self.probs
        sample_gold_prob = sample_gold_prob * torch.ones_like(prev_output_tokens[:, self.config.max_v_len:],
                                                              dtype=torch.float32)
        sample_gold_mask = torch.bernoulli(sample_gold_prob).long()

        updated_tokens = prev_output_tokens[:, self.config.max_v_len:] * sample_gold_mask + pred_tokens * (
                1 - sample_gold_mask)
        prev_output_tokens[:, self.config.max_v_len:] = updated_tokens

        return prev_output_tokens

    def _manage_scheduled_sampling(self, prediction_scores_dec1_pass1, input_ids):
        if self.config.scheduled_method == 'confidence':
            input_ids = self._update_scheduled_sampling(prediction_scores_dec1_pass1.detach(), input_ids)
        elif self.config.scheduled_method == 'probability':
            input_ids = self.get_word_orcale_tokens(prediction_scores_dec1_pass1.detach(), input_ids)
        else:
            raise ValueError("Unsupported Method {}".format(self.config.scheduled_method))

        return input_ids


    def _dec1_pass(self, input_ids, input_masks, token_type_ids, video_embeddings):
        embeddings_dec1 = self.embeddings_decs[0](input_ids, token_type_ids, video_embeddings=video_embeddings) # (N, L, D)

        dec1_outputs = self.decoders[0](
            embeddings_dec1, input_masks, output_all_encoded_layers=False)[-1]# both outputs are list
        
        prediction_scores_dec1 = self.pred_heads[0](dec1_outputs) # (S*B, L, vocab_size)

        return prediction_scores_dec1, dec1_outputs

    def forward_for_translate(self, query_clip, video_embeddings, input_ids, token_type_ids, input_masks,
                              sentence_embedding=None,
                              embeddings_layer=None, decoder_layer=None, prediction_head=None,
                              confidence_vector=None
                              ):
        embeddings = embeddings_layer(input_ids, token_type_ids, query_clip=query_clip,
                                      video_embeddings=video_embeddings,
                                      sentence_embeddings=sentence_embedding)  # (N, L, D)
        encoded_layer_outputs = decoder_layer(
            embeddings, input_masks, output_all_encoded_layers=False,
            confidence_vector=confidence_vector)  # both outputs are list
        prediction_scores = prediction_head(encoded_layer_outputs[-1])  # (N, L, vocab_size)

        return prediction_scores
    
    # * sentence embeddings
    # 将解码器输出的文本token嵌入聚合成句子级嵌入，支持三种聚合方式（均值、最大值、加权），用于后续迭代解码。
    def _meta_sen_emb_construction(self, metas, input_masks, dec1_outputs):
        B, S, D = metas
        sentence_embeddings = (input_masks.int().unsqueeze(-1) * dec1_outputs)[:, -(self.config.max_t_len):, :]

        if self.config.sentence_emb_aggregation_mode == 'mean':
            sentence_embeddings = torch.div(torch.sum(sentence_embeddings, dim=1),
                                            torch.sum(input_masks.int()[:, -(self.config.max_t_len):], dim=1).unsqueeze(
                                                -1))
        elif self.config.sentence_emb_aggregation_mode == 'max':
            sentence_embeddings = torch.max(sentence_embeddings, dim=1).values
        elif self.config.sentence_emb_aggregation_mode == 'weighted':
            # 加权聚合（通过MLP学习权重）
            _weighted = torch.softmax(self._weighted_para(sentence_embeddings), dim=1)
            sentence_embeddings = torch.bmm(_weighted.transpose(1, 2), sentence_embeddings)
            sentence_embeddings = sentence_embeddings.squeeze(1)
        else:
            raise ValueError("Unsupported Aggregation Mode {}".format(self.config.sentence_emb_aggregation_mode))
        
        # 调整形状为 (B, S, D)
        sentence_embeddings = sentence_embeddings.view(S, B, D).transpose(0, 1)
        return sentence_embeddings

    # 推理时构建句子嵌入和置信度向量：拼接多句子输入，通过解码器前向传播后，调用_conf_n_sen_emb_construction生成所需嵌入和向量。
    def construct_sentence_emb_for_translate(self, input_ids_list_prev, input_masks_list_prev, token_type_ids_list_prev,
                                             video_embeddings,
                                             embeddings_layer, decoder_layer, pred_head, prev_sentence_embeddings=None,
                                             prev_confidence_vector=None):
        input_ids, token_type_ids, input_masks = \
            _concat_list_of_tensors([input_ids_list_prev, token_type_ids_list_prev, input_masks_list_prev])
        
        embeddings_prev = embeddings_layer(input_ids, token_type_ids, video_embeddings=video_embeddings,
                                           sentence_embeddings=prev_sentence_embeddings)
        prev_outputs = decoder_layer(
            embeddings_prev, input_masks, confidence_vector=prev_confidence_vector, output_all_encoded_layers=False)[-1]
        prediction_scores_prev = pred_head(prev_outputs) # (S*B, L, vocab_size)

        B, S, D = input_ids_list_prev[0].shape[0], len(input_ids_list_prev), prev_outputs.shape[-1]
        metas = (B, S, D)

        sentence_embeddings, confidence_vector = self._conf_n_sen_emb_construction(metas, prediction_scores_prev,
                                                                                   prev_outputs, input_masks)
        return sentence_embeddings, prev_outputs, confidence_vector

    def _conf_n_sen_emb_construction(self, metas, prediction_scores_dec1, dec1_outputs, input_masks, n_S=1):
        B, S, D = metas

        _sentence_embeddings = self._meta_sen_emb_construction(metas, input_masks, dec1_outputs)
        sentence_embeddings = _sentence_embeddings.repeat(n_S, 1, 1)

        confidence_vector = self._meta_conf_vec_construction(metas, prediction_scores_dec1, _sentence_embeddings,
                                                             input_masks)
        confidence_vector = confidence_vector.repeat(n_S, 1)
        try:
            assert confidence_vector.max() < self.config.conf_bucket_size
            assert confidence_vector.min() >= 0
        except:
            confidence_vector = torch.clamp(confidence_vector, min=0, max=(self.config.conf_bucket_size - 1))

        # 构建句子嵌入矩阵（填充到指定长度：max_v_len + max_n_len + max_t_len）    
        N = (self.config.max_v_len + self.config.max_n_len + self.config.max_t_len)
        _sentence_embeddings = torch.zeros(n_S * B, N, self.config.hidden_size, dtype=sentence_embeddings.dtype,
                                           device=sentence_embeddings.device)
        _sentence_embeddings[:,
        self.config.max_v_len:self.config.max_v_len + self.config.max_n_len] = sentence_embeddings

        N = (self.config.max_v_len + self.config.max_n_len + self.config.max_t_len)
        assert confidence_vector.shape[1] == S
        if confidence_vector.shape[0] != (S * B):
            confidence_vector = confidence_vector.repeat(S, 1)
        _confidence_vector = confidence_vector.view(S, B, S)[0].view(B, S, 1)
        _confidence_vector = _confidence_vector.repeat(1, 1, N)
        _confidence_vector = _confidence_vector.transpose(0, 1)
        _confidence_vector = _confidence_vector.reshape(-1, N)
        _confidence_vector[:, self.config.max_v_len:self.config.max_v_len + self.config.max_n_len] = confidence_vector
        assert _confidence_vector.max() < self.config.conf_bucket_size
        assert _confidence_vector.min() >= 0
        return _sentence_embeddings, _confidence_vector

    @torch.no_grad()
    def _meta_conf_vec_construction(self, metas, prediction_scores_dec1, senten_emb, input_masks):
        confidence_vector = self._generate_conf_matrix_pred(metas, prediction_scores_dec1, input_masks)
        return confidence_vector.detach()

    @torch.no_grad()
    def _generate_conf_matrix_pred(self, metas, pred, input_masks):
        temprature = self.config.conf_temperature
        B, S, _ = metas
        _word_score = torch.max(pred, dim=-1).values
        _word_score = (input_masks.int() * _word_score)[:, -(self.config.max_t_len):]
        _sen_score = torch.div(torch.sum(_word_score, dim=1),
                               torch.sum(input_masks.int()[:, -(self.config.max_t_len):], dim=1))
        _sen_score = _sen_score.view(S, B).transpose(0, 1)
        _sen_score = torch.softmax(_sen_score / temprature, dim=1)

        # * quantize
        _sen_score = (_sen_score * self.conf_bucket_size).floor().long()
        return _sen_score

    @torch.no_grad()
    def _momentum_update_aux_encoder(self):
        """
               Momentum update of the aux encoder
        """
        for param_q, param_k in zip(self.video_encoder.parameters(), self.video_encoder_aux.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projection_head_q.parameters(), self.projection_head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    # --- [新增] 序列长度对齐辅助函数 ---
    def _align_sequence_length(self, student_logits, teacher_logits, valid_mask=None):
        """
        处理 student (BS, Ls, V) 和 teacher (BS, Lt, V) 之间的长度差异
        如果维度是 (BS, V)，则直接返回（不处理）
        """
        # 1. 维度标准化: 如果教师logits是2D (BS, V)，先扩展为3D (BS, 1, V)
        if teacher_logits.dim() == 2:
            teacher_logits = teacher_logits.unsqueeze(1)

        if student_logits.dim() != 3 or teacher_logits.dim() != 3:
            return teacher_logits

        s_len = student_logits.size(1)
        t_len = teacher_logits.size(1)

        if s_len == t_len:
            return teacher_logits

        # 截断
        if t_len > s_len:
            return teacher_logits[:, :s_len, :]

        # 填充 (使用极小值避免影响 Softmax)
        if t_len < s_len:
            # [核心修复] 特殊情况处理：
            # 如果教师只有1个token（通常是Pass 1的全局事件分类），而学生在生成序列，
            # 我们应该将教师的这个token“广播”到学生的所有位置，作为全局上下文约束。
            # 否则，填充-inf会导致有效label位置对应的教师信号为空，导致Loss=0
            if t_len == 1:
                return teacher_logits.expand(-1, s_len, -1)
            
            if valid_mask is not None and self.training:
                # valid_mask shape: (BS*Ls) -> view as (BS, Ls)
                # 检查 s_len 中超出 t_len 的部分是否有有效标签
                extra_mask = valid_mask.view(student_logits.size(0), s_len)[:, t_len:]
                if extra_mask.sum() > 0:
                    # 这意味着我们在用 "空" 蒸馏 "实"
                    pass

            pad_size = s_len - t_len
            padding = torch.full(
                (teacher_logits.size(0), pad_size, teacher_logits.size(2)),
                fill_value=-1e18,
                dtype=teacher_logits.dtype,
                device=teacher_logits.device
            )
            return torch.cat([teacher_logits, padding], dim=1)

    def _forward_refactored(self,
                            encoder_input, unmasked_encoder_input,
                            input_ids_list, input_masks_list, token_type_ids_list,
                            input_labels_list=None,
                            teacher_logits=None
                            ):

        aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        loss_iikd = torch.tensor(0.0, device=next(self.parameters()).device)
        caption_loss = 0.
        prediction_scores_list = []

        # * masked forward
        # 1. 主编码器（VideoEncodingTrans）前向传播，生成带因果信息的视频事件特征
        video_embeddings = self.video_encoder(**encoder_input)

        # aux video loss
        # 2. 主编码器输出投影（对齐辅助编码器输出维度）
        _video_embeddings = self.projection_head_q(video_embeddings)

        with torch.no_grad():
            self._momentum_update_aux_encoder()
            unmasked_video_embeddings = self.video_encoder_aux(**unmasked_encoder_input)
            _unmasked_video_embeddings = self.projection_head_k(unmasked_video_embeddings)

            _unmasked_video_embeddings = _unmasked_video_embeddings * unmasked_encoder_input['video_mask'][..., None]
            _unmasked_video_embeddings = _unmasked_video_embeddings.detach_()

        # * only unmasked part need to calculate loss
        _video_embeddings = _video_embeddings * unmasked_encoder_input['video_mask'][..., None]

        # FIX 2:就是原VAR中的aux_loss
        aux_loss = F.mse_loss(_video_embeddings,
                                      _unmasked_video_embeddings,
                                      reduction='sum')
        aux_loss = aux_loss / (torch.sum(unmasked_encoder_input['video_mask']).detach())
        caption_loss += aux_loss

        # --- 新增: 准备教师 Logits ---
        final_teacher_logits_list = []
        is_distilling = self.training and self.config.use_distillation and teacher_logits is not None

        if is_distilling:
            # teacher_logits 形状 (B, S, K, L, V)
            # 我们需要将其拆解为 K 个列表。
            # 注意：var_dataset 中 k=0 放在 [:, :, 0, 0, :]，形状 (B, S, 1, V) -> (B, S, V)
            # k>0 放在 [:, :, k, :, :]，形状 (B, S, L, V)

            for k in range(self.config.K):
                if k == 0:
                    # Pass 1: 取出 (B, S, V)
                    # teacher_logits[:, :, 0, 0, :]
                    logits_k = teacher_logits[:, :, 0, 0, :]
                    # 展平为 (B*S, V)
                    flattened_logits_k = logits_k.contiguous().view(-1, logits_k.shape[-1])
                else:
                    # Pass 2+: 取出 (B, S, L, V)
                    logits_k = teacher_logits[:, :, k, :, :]
                    # 展平为 (B*S, L, V)
                    flattened_logits_k = logits_k.contiguous().view(-1, logits_k.shape[-2], logits_k.shape[-1])

                final_teacher_logits_list.append(flattened_logits_k)

            if len(final_teacher_logits_list) != self.config.K:
                logging.error(f"Teacher Logits List size mismatch! Disabling.")
                is_distilling = False

        B = video_embeddings.shape[0]
        S = len(input_ids_list[0])
        # 2. 解码器 Pass 1 (K=0)
        input_labels, input_ids, token_type_ids, input_masks = \
            _concat_list_of_tensors(
                [input_labels_list[0], input_ids_list[0], token_type_ids_list[0], input_masks_list[0]])

        if not self.config.disable_scheduled_sampling and self.training and hasattr(self, 'epoch') and self.epoch >= 3:
            with torch.no_grad():
                prediction_scores_dec1_pass1, _ = self._dec1_pass(input_ids, input_masks, token_type_ids,
                                                                  video_embeddings)
                input_ids = self._manage_scheduled_sampling(prediction_scores_dec1_pass1, input_ids)

                                                              
        prediction_scores_dec1, dec1_outputs = self._dec1_pass(input_ids, input_masks, token_type_ids,
                                                                video_embeddings)

        D = dec1_outputs.shape[-1]
        metas = (B, S, D)
        prev_outputs = dec1_outputs
        prev_prediction_scores = prediction_scores_dec1

        total_aux_caption_loss = torch.tensor(0.0, device=video_embeddings.device)

        # 模型包含 K 个解码器（decoders），Intermediate Decoders Loop (k=1 to K-1)
        for k in range(1, self.config.K):
            # 计算前一步 (k-1) 的损失 (注意：k=1时，前一步是 Pass 1，我们已经在上面算过了)
            # 原 VAR 逻辑：loop k=1..K. k<K 计算 loss_aux.
            # 这里的 input_labels 是上一轮的 input_labels。
            # 如果 k=1, input_labels 还是 input_labels_list[0].
            # prev_prediction_scores 是 prediction_scores_dec1.
            # 这意味着 loop 中的第一次迭代 (k=1) 计算的其实是 Pass 1 的损失。
            # 所以上面的 "Pass 1 损失计算" 代码块其实应该在 loop 内部处理。
            # 为了保持原逻辑结构不被破坏，我们按照原 mainmodel 逻辑：
            # Loop 从 1 到 K (不含K+1? 你的代码是 range(1, config.K))

            # --- [修改] Loop 内损失计算 (Hard + Distill) ---
            # 这里的 loss 是针对 prev_prediction_scores (即第 k-1 层输出)

            # 注意：如果你上面已经加了 combined_loss_aux0，这里 k=1 时会重复计算 Pass 1 的损失。
            # 原代码逻辑：total_aux_caption_loss 在 loop 内累加。
            # 所以，我应该移除上面的 "Pass 1 损失累加"，完全在 Loop 内处理。
            # 但是 Loop 只处理到 K-1。如果是 K=4，Loop 1,2,3。
            # k=1: 算 Pass 1 损失。
            # k=2: 算 Pass 2 损失。
            # k=3: 算 Pass 3 损失。
            # Main Loss: 算 Pass 4 (K) 损失。
            # 这样是闭环的。所以上面的 "Pass 1 损失计算" 代码块可以删除或注释，统一在 Loop 内。

            # 但为了稳妥，且你的代码结构里 dec1_pass 后紧接 Loop，
            # 我将 Loop 内的损失计算进行增强。
            k_minus_1_scores = prev_prediction_scores
            current_hard_labels = input_labels

            loss_hard_k = self.hard_loss_fn(
                k_minus_1_scores.view(-1, self.config.vocab_size),
                current_hard_labels.view(-1)
            )
            combined_loss_k = loss_hard_k

            if is_distilling:
                teacher_logits_k = final_teacher_logits_list[k - 1]

                # [CRITICAL FIX 4] 传入 valid_mask 辅助对齐决策
                # 这里的 current_hard_labels 包含了 padding (-1)，我们需要它的 bool mask
                valid_mask_k = (current_hard_labels != -1)

                teacher_logits_k = self._align_sequence_length(k_minus_1_scores, 
                                                               teacher_logits_k, 
                                                               valid_mask=valid_mask_k)

                loss_iikd_k, loss_sa_k, loss_sd_k = self.distill_loss_fn(
                    k_minus_1_scores, current_hard_labels, teacher_logits_k
                )
                
                combined_loss_k = self.config.delta_hard_loss * loss_hard_k + \
                                  self.config.gamma_distill_loss * loss_iikd_k
                '''
                loss_iikd += self.config.loss_aux_caption * loss_iikd_k
                '''
            # 累加到 total_aux_caption_loss
            total_aux_caption_loss += self.config.loss_aux_caption * combined_loss_k
            caption_loss += self.config.loss_aux_caption * combined_loss_k
            '''
            print("-" * 60)
            print(f"    - Intermediate Decoder hard {k} Loss: {loss_hard_k.item():.6f}")
            print(f"    - Intermediate Decoder {k} Loss: {combined_loss_k.item():.6f}")
            #print(f"    - iikd Loss: {loss_iikd.item():.6f}")
            print(f"    - Distillation Loss Components (unweighted):")
            print(f"    - L_iikd (total):    {loss_iikd_k.item():.6f}")
            print(f"    - L_sa (Target-BCE): {loss_sa_k.item():.6f}")
            print(f"    - L_sd (Non-Target-MSE): {loss_sd_k.item():.6f}")
            print(f"    - L_sa_weight (Target-BCE): {loss_sa_k.item() * self.config.alpha_sa_loss:.6f}")
            print(f"    - L_sd_weight (Non-Target-MSE): {loss_sd_k.item() * self.config.beta_sd_loss:.6f}")
            print("-" * 60)
            '''
            # --- 运行当前解码器 k ---
            sentence_embeddings, confidence_vector = self._conf_n_sen_emb_construction(metas, prev_prediction_scores,
                                                                                       prev_outputs, input_masks, n_S=S)
            # * dec2
            input_labels, input_ids, token_type_ids, input_masks = \
                _concat_list_of_tensors([input_labels_list[k], input_ids_list[k], token_type_ids_list[k], input_masks_list[k]])

            prev_pred_id = prev_prediction_scores.max(2).values  # S*B, L
            input_ids[:, -self.config.max_t_len] = prev_pred_id[:, -self.config.max_t_len].detach()

            _video_embeddings = prev_outputs[:, :(self.config.max_v_len), :]

            embeddings_dec2 = self.embeddings_decs[k](input_ids, token_type_ids,
                                                       video_embeddings=_video_embeddings,
                                                       sentence_embeddings=sentence_embeddings)

            current_outputs = self.decoders[k](embeddings_dec2, input_masks, output_all_encoded_layers=False,
                                               confidence_vector=confidence_vector)[-1]
            # * final pred
            current_prediction_scores = self.pred_heads[k](current_outputs)
            prediction_scores_list = current_prediction_scores.view(S,B,*current_prediction_scores.shape[1:])

            # * for next decoder
            prev_outputs = current_outputs
            prev_prediction_scores = current_prediction_scores

        # --- 最终损失计算 ---
        if self.config.K == 1:
            current_prediction_scores = prediction_scores_dec1
            prediction_scores_list = current_prediction_scores.view(S,B,*current_prediction_scores.shape[1:])
            final_hard_labels = _concat_list_of_tensors([input_labels_list[0]])[0]
        else:
            final_hard_labels = input_labels

        # --- 修改: 组合最终硬损失和蒸馏损失 ---
        loss_hard_main = self.hard_loss_fn(
            current_prediction_scores.view(-1, self.config.vocab_size),
            final_hard_labels.view(-1)
        )
        combined_loss_main = loss_hard_main  # 默认值

        if is_distilling:
            # 使用最后一个 (K-1) 教师 Logit
            teacher_logits_main = final_teacher_logits_list[self.config.K - 1]
            # [CRITICAL FIX 5] 同样的对齐逻辑应用到主损失
            valid_mask_main = (final_hard_labels != -1)
            teacher_logits_main = self._align_sequence_length(current_prediction_scores, 
                                                              teacher_logits_main, 
                                                              valid_mask=valid_mask_main)

            loss_iikd_main, _, _ = self.distill_loss_fn(
                current_prediction_scores, final_hard_labels, teacher_logits_main
            )
            # 组合损失
            
            combined_loss_main = self.config.delta_hard_loss * loss_hard_main + \
                                 self.config.gamma_distill_loss * loss_iikd_main
            '''
            combined_loss_main = self.config.loss_main_caption * loss_hard_main
            loss_iikd += self.config.loss_main_caption * loss_iikd_main
            '''                  
        main_caption_loss = combined_loss_main

        # 正确代码
        #caption_loss = (self.config.loss_aux_weight * aux_loss) + self.config.delta_hard_loss * (main_caption_loss + total_aux_caption_loss) + self.config.gamma_distill_loss * loss_iikd
        caption_loss += self.config.loss_main_caption * main_caption_loss 

        # [修改 3] 增加计数器累加，并设置每 500 轮打印一次
        if self.training:
            self.current_step_counter += 1

            if self.current_step_counter == 1 or self.current_step_counter % 500 == 0:
                print("-" * 60)
                print(f"[Epoch {self.epoch} | Step {self.current_step_counter}] LOSS DIAGNOSIS")
                print(f"    - Original VAR Loss Components:")
                print(f"    - aux_loss : {aux_loss.item():.6f}")
                print(f"    - Intermediate Decoders sum : {total_aux_caption_loss.item() if isinstance(total_aux_caption_loss, torch.Tensor)else total_aux_caption_loss:.6f}")
                print(f"    - Main Decoder hard Loss : {loss_hard_main.item():.6f}")
                print(f"    - Final Decoder CE : {combined_loss_main.item():.6f}")         
                if is_distilling:
                    print(f"    - Main Decoder iikd Loss : {loss_iikd_main.item():.6f}")
                    #print(f"    - Total iikd Loss : {loss_iikd.item():.6f}")
                print(f"    - Total Caption Loss: {caption_loss.item():.6f}")
                print("-" * 60)
        '''
        print("-" * 60)
        print("[LOSS DIAGNOSIS]")
        print(f"    - Original VAR Loss Components:")
        print(f"    - aux_loss : {aux_loss.item():.6f}")
        print(f"    - Intermediate Decoders sum : {total_aux_caption_loss.item() if isinstance(total_aux_caption_loss, torch.Tensor)else total_aux_caption_loss:.6f}")
        print(f"    - Main Decoder hard Loss : {loss_hard_main.item():.6f}")
        print(f"    - Final Decoder CE : {combined_loss_main.item():.6f}")         
        print(f"    - Main Decoder iikd Loss : {loss_iikd_main.item():.6f}")
        #print(f"    - Total iikd Loss : {loss_iikd.item():.6f}")
        print(f"    - Total Caption Loss: {caption_loss.item():.6f}")
        print("-" * 60)
        '''
        # --- START OF MODIFICATION: 将用于性能计算的张量保存为属性 ---
        # 保存这两个张量，以便在 train_epoch 中访问它们来计算准确率
        self.last_pred_scores = current_prediction_scores
        self.last_gold_labels = final_hard_labels
        # --- END OF MODIFICATION ---

        return caption_loss, prediction_scores_list

    def forward(self, **kwargs):
        #total_loss = self._forward_refactored(**kwargs)
        #return total_loss
        return self._forward_refactored(**kwargs)
        