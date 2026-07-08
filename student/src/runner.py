'''
This script handles the training process for the student model.
'''
'''
核心改动:
get_args(): 增加知识蒸馏相关的命令行参数（--use_distillation, --soft_label_path, alpha, beta, temperature）。移除教师模型专用的--model_mode参数。
main(): 在初始化VARDataset时，传入soft_label_path。在分布式训练中，加载软标签文件。
train_logic(): (通过 train_epoch 间接修改) 将软标签传递给模型。
'''
'''
get_args: 参数名仍然是soft_label_path，但其含义和默认值已更新为指向一个目录。
main: 传递给VARDataset的关键字参数现在是soft_label_dir，与VARDataset的__init__方法匹配。
'''
import os
import math
import time
import json
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import get_model
from model.utils.optimization import BertAdam, EMA
from dataloader.var_dataset import VARDataset, sentences_collate

from engine.train import train_epoch
from engine.valid import eval_epoch
from engine.translate import translate_w_metrics

from utils.io import save_parsed_args_to_json
from utils.dist import is_distributed, dist_log
from utils.misc import init_seed, create_save_dir
import numpy as np


class runner():

    @staticmethod
    def train_logic(epoch_i, model, training_data, optimizer, ema, device, opt):
        if is_distributed():
            training_data.sampler.set_epoch(epoch_i)
            dist_log('Setting sampler seed: {}'.format(training_data.sampler.epoch))

        start = time.time()
        if ema is not None and epoch_i != 0:
            ema.resume(model)
        train_loss, train_acc = train_epoch(
            model, training_data, optimizer, ema, device, opt, epoch_i)

        dist_log('[Training]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, elapse {elapse:3.3f} min'
                    .format(ppl=math.exp(min(train_loss, 100)), acc=100*train_acc, elapse=(time.time()-start)/60.))

        if ema is not None:
            ema.assign(model)

        return model, ema, train_loss, train_acc

    @staticmethod
    def eval_logic(model, validation_data, device, opt):
        start = time.time()
        val_loss, val_acc = eval_epoch(model, validation_data, device, opt)
        dist_log('[Val]  ppl: {ppl: 8.5f}, accuracy: {acc:3.3f} %, elapse {elapse:3.3f} min'
                    .format(ppl=math.exp(min(val_loss, 100)), acc=100*val_acc, elapse=(time.time()-start)/60.))
        return model, val_loss, val_acc

    @staticmethod
    def translate_logic(epoch_i, model, translation_data, opt, prev_best_score):
        if hasattr(model, 'module'):
            _model = model.module
        else: _model = model

        checkpoint = {
            'model': _model.state_dict(),
            'model_cfg': _model.config,
            'epoch': epoch_i}

        val_greedy_output, filepaths = translate_w_metrics(
                checkpoint, translation_data, opt, eval_mode=opt.evaluate_mode, model=_model)

        if opt.local_rank <= 0:
            if 'BertScore' not in val_greedy_output:
                val_greedy_output['BertScore'] = [0.]

            dist_log('[Val] METEOR {m:.2f} Bleu@4 {b:.2f} CIDEr {c:.2f} ROUGE_L {r:.2f} BERT_S {s:.2f}'
                        .format(m=val_greedy_output['METEOR'][0]*100,
                                b=val_greedy_output['Bleu_4'][0]*100,
                                c=val_greedy_output['CIDEr'][0]*100,
                                r=val_greedy_output['ROUGE_L'][0]*100,
                                s=val_greedy_output['BertScore'][0]*100))

            if opt.save_mode == 'all':
                model_name = opt.save_model + '_e{}.chkpt'.format(epoch_i)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if val_greedy_output['CIDEr'][0] > prev_best_score:
                    prev_best_score = val_greedy_output['CIDEr'][0]
                    torch.save(checkpoint, model_name)
                    # 修改后的代码
                    new_filepaths = [e.replace('tmp', 'best') for e in filepaths]
                    for src, tgt in zip(filepaths, new_filepaths):
                        if os.path.exists(tgt):
                            os.remove(tgt)  # 在重命名前删除已存在的目标文件
                        os.renames(src, tgt)
                    dist_log('The checkpoint file has been updated.')

        return model, val_greedy_output, prev_best_score

    @staticmethod
    def run(model, training_data, validation_data, translation_data, device, opt):
        param_optimizer = list(model.named_parameters())

        if opt.use_shared_txt_emb:
            if hasattr(model, 'module'):
                for parameter in model.module.embeddings_dec2.txt_emb.parameters():
                    parameter.requires_grad = False
            else:
                for parameter in model.embeddings_dec2.txt_emb.parameters():
                    parameter.requires_grad = False

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if (p.requires_grad and (not any(nd in n for nd in no_decay)))],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if (p.requires_grad and any(nd in n for nd in no_decay))],
             'weight_decay': 0.0}
        ]

        ema = EMA(opt.ema_decay) if opt.ema_decay != -1 else None
        if ema:
            for name, p in model.named_parameters():
                if p.requires_grad:
                    ema.register(name, p.data)

        num_train_optimization_steps = len(training_data) * opt.n_epoch
        optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=opt.lr,
                            warmup=opt.lr_warmup_proportion,
                            t_total=num_train_optimization_steps,
                            schedule='warmup_linear')

        log_train_file, log_valid_file = None, None
        if opt.log and opt.local_rank <= 0:
            log_train_file = opt.log + '.train.log'
            log_valid_file = opt.log + '.valid.log'
            dist_log(f'Training logs will be written to: {log_train_file} and {log_valid_file}')
            with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
                log_tf.write('epoch,loss,ppl,accuracy\n')
                log_vf.write('epoch,loss,ppl,accuracy,METEOR,BLEU@4,CIDEr,ROUGE\n')

        # --- Mod Start: 权重初始化诊断 & 梯度裁剪确认 ---
        # 针对你怀疑参数未初始化的修改：在训练开始前打印关键层权重的统计信息
        # 如果 Std 远大于 0.02 (如 ~1.0) 或为 0，说明初始化有问题。正常应接近 opt.initializer_range (0.02)
        if opt.local_rank <= 0:
            dist_log("="*20 + " [DIAGNOSIS] Parameter Initialization Check " + "="*20)
            check_layers = ['video_encoder', 'decoders.0', 'projection_head'] # 抽样检查关键层
            count = 0
            for name, param in model.named_parameters():
                if any(k in name for k in check_layers) and 'weight' in name and param.dim() > 1:
                    dist_log(f"Layer: {name} | Mean: {param.data.mean():.6f} | Std: {param.data.std():.6f}")
                    count += 1
                    if count >= 6: break # 只打印前几个，避免刷屏
            dist_log(f"Expected Std should be close to {opt.initializer_range}. If Std >> 0.02, initialization failed.")
            
            dist_log("-" * 60)
            if opt.grad_clip > 0:
                dist_log(f"[CONFIG CHECK] Gradient Clipping ENABLED: max_norm={opt.grad_clip}")
            else:
                dist_log(f"[CONFIG CHECK] WARNING: Gradient Clipping DISABLED (grad_clip={opt.grad_clip}). Transformer training may be unstable.")
            dist_log("="*80)
        # --- Mod End ---

        prev_best_score = 0.
        for epoch_i in range(opt.n_epoch):
            dist_log(f'[Epoch {epoch_i}]')

            model, ema, train_loss, train_acc = \
                    runner.train_logic(epoch_i, model, training_data, optimizer, ema, device, opt)
            model, val_loss, val_acc = runner.eval_logic(model, validation_data, device, opt)

            if epoch_i >= opt.trans_sta_epoch:
                model, val_greedy_output, prev_best_score = \
                    runner.translate_logic(epoch_i, model, translation_data, opt, prev_best_score)

                if opt.local_rank <= 0:
                    cfg_name = opt.save_model +'.cfg.json'
                    save_parsed_args_to_json(opt, cfg_name)
                    if log_train_file and log_valid_file:
                        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{acc:3.3f}\n'.format(
                                epoch=epoch_i, loss=train_loss, ppl=math.exp(min(train_loss, 100)),
                                acc=100 * train_acc))
                            log_vf.write(
                                '{epoch},{loss: 8.5f},{ppl: 8.5f},{acc:3.3f},{m:.2f},{b:.2f},{c:.2f},{r:.2f},{s:.2f}\n'.format(
                                    epoch=epoch_i, loss=val_loss, ppl=math.exp(min(val_loss, 100)), acc=100 * val_acc,
                                    m=val_greedy_output['METEOR'][0] * 100,
                                    b=val_greedy_output['Bleu_4'][0] * 100,
                                    c=val_greedy_output['CIDEr'][0] * 100,
                                    r=val_greedy_output['ROUGE_L'][0] * 100,
                                    s=val_greedy_output['BertScore'][0] * 100
                                ))
        return model


def get_args():
    '''parse and preprocess cmd line args'''
    parser = argparse.ArgumentParser()

    # ... (所有原始参数保持不变) ...
    parser.add_argument('--dset_name', type=str, default='VAR', choices=['VAR'])
    parser.add_argument('--model_name', type=str, default='ReasonerStudent')
    parser.add_argument('--data_dir', required=True, help='Path to the main dataset directory (e.g., .../my/data/VAR).')
    parser.add_argument('--res_root_dir', type=str, default='results')
    parser.add_argument('--data_list', type=str, default='results')

    # * training config -- batch/lr/eval etc.
    parser.add_argument('--n_epoch', type=int, default=17, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=16, help='inference batch size')
    parser.add_argument('--trans_batch_size', type=int, default=16, help='tranlating batch size')
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--lr_warmup_proportion', default=0.1, type=float,
                        help='Proportion of training to perform linear learning rate warmup for. '
                             'E.g., 0.1 = 10% of training.')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Use soft target instead of one-hot hard target')
    parser.add_argument('--grad_clip', type=float, default=1, help='clip gradient, -1 == disable')
    parser.add_argument('--ema_decay', default=0.9999, type=float,
                        help='Use exponential moving average at training, float in (0, 1) and -1: do not use.  '
                             'ema_param = new_param * ema_decay + (1-ema_decay) * last_param')

    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Don\'t use pin_memory=True for dataloader. '
                             'ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num subprocesses used to load the data, 0: use main process')
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best',
                        help='all: save models at each epoch; best: only save the best model')
    parser.add_argument('--no_cuda', action='store_true', help='run on cpu')

    parser.add_argument('--evaluate_mode', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--score_mode', type=str, default='event', choices=['event'])
    parser.add_argument('--trans_sta_epoch', type=int, default=2)

    # * model overall config
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--intermediate_size', type=int, default=1152)# 768*1.5
    parser.add_argument('--vocab_size', type=int, help='number of words in the vocabulary')
    parser.add_argument('--word_vec_size', type=int, default=300)
    parser.add_argument('--video_feature_size', type=int, default=3072, help='2048 appearance + 1024 flow')
    parser.add_argument('--max_v_len', type=int, default=50, help='max length of video feature')
    parser.add_argument('--max_t_len', type=int, default=22,
                        help='max length of text (sentence or paragraph)')
    parser.add_argument('--max_n_len', type=int, default=12,
                        help='for recurrent, max number of sentences')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='')

    # * initialization config
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='number of transformer layers')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.1)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--initializer_range', type=float, default=0.02)

    # * visual encoder config
    parser.add_argument('--K', type=int, default=4, help='number of cascades')
    parser.add_argument('--loss_aux_weight', type=float, default=0.1, help='')
    parser.add_argument('--momentum_aux_m', type=float, default=0.999)

    # * linguistic encoder config
    parser.add_argument('--glove_path', type=str, default=None, help='Path to the directory containing GloVe file.')
    parser.add_argument('--glove_version', type=str, default=None, help='extracted GloVe vectors')
    parser.add_argument('--freeze_glove', action='store_true', help='do not train GloVe vectors')
    parser.add_argument('--share_wd_cls_weight', action='store_true',
                        help='share weight matrix of the word embedding with the final classifier, ')

    # * cascade decoder config
    parser.add_argument('--num_dec1_blocks', type=int, default=4)
    parser.add_argument('--num_dec2_blocks', type=int, default=4)
    parser.add_argument('--loss_aux_caption', type=float, default=0.25)
    parser.add_argument('--loss_main_caption', type=float, default=0.25)
    parser.add_argument('--use_shared_txt_emb', action='store_true')
    # * scheduled sampling
    parser.add_argument('--disable_scheduled_sampling', action='store_true')
    parser.add_argument('--scheduled_method', type=str, default='probability', choices=['confidence', 'probability'])
    parser.add_argument('--conf_replace_tr', type=float, default=0.5)
    parser.add_argument('--decay_k', type=int, default=10)
    # * sentence embedding and conf hyper-parameters
    parser.add_argument('--sentence_emb_aggregation_mode', type=str, default='max', choices=['mean', 'max', 'weighted'])
    parser.add_argument('--conf_bucket_size', type=int, default=10)
    parser.add_argument('--conf_temperature', type=float, default=1.0)

    # --- 核心修改点：添加知识蒸馏相关参数 ---
    parser.add_argument('--use_distillation', action='store_true',
                        help='Enable knowledge distillation training mode.')
    # --- 核心修改点：更新软标签参数的帮助文本和默认值 ---
    parser.add_argument('--soft_label_path', type=str,
                        help='Path to the DIRECTORY containing soft_labels.pkl and index.')
    parser.add_argument('--alpha_sa_loss', type=float, default=1.0,
                        help='Weight for the SAKD loss in KD.')
    parser.add_argument('--beta_sd_loss', type=float, default=300.0,
                        help='Weight for the  SDKD label (distillation) loss in KD.')
    parser.add_argument('--temperature_kd', type=float, default=2.0,
                        help='Temperature for softening probability distributions in KD.')
    parser.add_argument('--gamma_distill_loss', type=float, default=500.0, help='')
    parser.add_argument('--delta_hard_loss', type=float, default=0.8, help='')
    #parser.add_argument('--bert_score_model_path', type=str, default=None, help='Path to the local directory of the roberta-large model for offline BertScore calculation.')


    # --- 修改结束 ---

    opt = parser.parse_args()
    opt.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    opt.cuda = not opt.no_cuda
    opt.pin_memory = not opt.no_pin_memory

    if opt.share_wd_cls_weight:
        assert opt.word_vec_size == opt.hidden_size
    if opt.K == 1:
        opt.loss_main_caption = 1.
        opt.loss_aux_caption = 0.
        opt.disable_scheduled_sampling = True

    return opt


def main():
    opt = get_args()

    # --- 核心修改点：路径规范化 ---
    # 将从bash脚本接收的、可能格式混乱的路径，转换为Python和当前操作系统认可的绝对路径
    opt.data_dir = os.path.abspath(opt.data_dir)
    if opt.glove_path:
        opt.glove_path = os.path.abspath(opt.glove_path)
    if opt.soft_label_path:
        opt.soft_label_path = os.path.abspath(opt.soft_label_path)
    #if opt.bert_score_model_path:
        #opt.bert_score_model_path = os.path.abspath(opt.bert_score_model_path)
    # --- 修改结束 ---

    init_seed(opt.seed, cuda_deterministic=True)

    distributed = 'LOCAL_RANK' in os.environ and int(os.environ['LOCAL_RANK']) >= 0

    # --- 核心修改点：为学生模型加载数据集 ---
    # 学生模型总是使用 'student' 模式的数据集，即带有掩码的
    train_dataset = VARDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir,
        max_t_len=opt.max_t_len, max_v_len=opt.max_v_len, max_n_len=opt.max_n_len,
        mode='train', K=opt.K,
        word2idx_path=os.path.join(opt.glove_path, opt.glove_version.replace('vocab_glove', 'word2idx').replace('.pt', '.json')),
        soft_label_dir=opt.soft_label_path if opt.use_distillation else None
    )
    # 验证/测试集不需要软标签
    val_dataset = VARDataset(
        dset_name=opt.dset_name,
        data_dir=opt.data_dir,
        max_t_len=opt.max_t_len, max_v_len=opt.max_v_len, max_n_len=opt.max_n_len,
        mode=opt.evaluate_mode, K=opt.K,
        word2idx_path=os.path.join(opt.glove_path, opt.glove_version.replace('vocab_glove', 'word2idx').replace('.pt', '.json'))
    )
    # --- 修改结束 ---

    if distributed:
        device = torch.device(f'cuda:{opt.local_rank}')
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        device = torch.device('cuda:0' if opt.cuda and torch.cuda.is_available() else 'cpu')

    train_sampler, val_sampler = (DistributedSampler(train_dataset), DistributedSampler(val_dataset)) if distributed else (None, None)

    opt = create_save_dir(opt)

    train_loader = DataLoader(train_dataset, collate_fn=sentences_collate,
                              batch_size=opt.batch_size, shuffle=(train_sampler is None),
                              num_workers=opt.num_workers, pin_memory=opt.pin_memory, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, collate_fn=sentences_collate,
                            batch_size=opt.val_batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=opt.pin_memory, sampler=val_sampler)
    trans_loader = DataLoader(val_dataset, collate_fn=sentences_collate,
                            batch_size=opt.trans_batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=opt.pin_memory, sampler=val_sampler)

    opt.vocab_size = len(train_dataset.word2idx)
    if opt.local_rank <= 0: print(json.dumps(vars(opt), indent=4, sort_keys=True))

    model = get_model(opt)

    if opt.glove_path is not None:
        if hasattr(model, 'embeddings_decs'):
            dist_log('Load GloVe as word embedding')
            glove_file = os.path.join(opt.glove_path, opt.glove_version)
            if os.path.exists(glove_file):
                # --- 核心修改点：添加 weights_only=False ---
                # 加载包含numpy对象的旧格式pt文件时，需要禁用纯权重加载模式
                glove_embeddings = torch.load(glove_file, weights_only=False)
                # --- 修改结束 ---

                # 有时加载出来直接是tensor，有时是numpy，做一下兼容
                if isinstance(glove_embeddings, np.ndarray):
                    glove_embeddings = torch.from_numpy(glove_embeddings)

                for k in range(opt.K):
                    # 确保模型已经移动到设备上
                    if hasattr(model, 'module'):
                        model.module.embeddings_decs[k].txt_emb.set_pretrained_embedding(
                            glove_embeddings.float(), freeze=opt.freeze_glove)
                    else:
                        model.embeddings_decs[k].txt_emb.set_pretrained_embedding(
                            glove_embeddings.float(), freeze=opt.freeze_glove)
            else:
                dist_log(f'[WARNING] GloVe file not found at: {glove_file}')
        else:
            dist_log('[WARNING] This model has no attribute "embeddings_decs", cannot load glove vectors.')

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True, device_ids=[opt.local_rank], output_device=opt.local_rank)
    else:
        model = model.to(device)

    runner().run(model, train_loader, val_loader, trans_loader, device, opt)
    print(f"Train dataset size: {len(train_dataset)}")  # 添加数据集大小打印
    print(f"Val dataset size: {len(val_dataset)}")  # 添加验证集大小打印

    # 添加空数据集检查
    if len(train_dataset) == 0:
        raise ValueError("训练数据集为空！请检查数据路径和特征文件")


if __name__ == '__main__':
    main()