import os
import math
import nltk
import numpy as np
import torch
import json
import pickle

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from utils import load_json, flat_list_of_lists

import logging
logger = logging.getLogger(__name__)
'''
内存高效的软标签加载：
__init__方法接收一个 soft_label_dir 目录路径。
在该目录中加载 soft_labels_index.json 索引文件到内存。
以二进制只读模式打开 soft_labels.pkl 文件，并仅保留其文件句柄，不加载内容到RAM。
__getitem__ 方法根据索引文件中的字节偏移量(offset)和长度(length)，按需从.pkl文件句柄中seek并read单个样本的数据，实现近乎零开销的按需加载，彻底解决 MemoryError。
添加 __del__ 方法以确保在程序退出时文件句柄被正确关闭。
'''
log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


class VARDataset(Dataset):
    """
    VARDataset for Student Model.
    - Always masks the hypothesis (student mode).
    - If soft_label_dir is provided (training mode), it loads teacher logits on-the-fly
      using an index file and binary file seeking to keep memory usage minimal.
    """
    PAD_TOKEN = "[PAD]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    VID_TOKEN = "[VID]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    SEN_TOKEN = "[SEN]"
    MSK_TOKEN = "[MSK]"

    PAD = 0
    CLS = 1
    SEP = 2
    VID = 3
    BOS = 4
    EOS = 5
    UNK = 6
    SEN = 7
    MSK = 8
    IGNORE = -1

    VID_TYPE = 0
    TEX_TYPE = 1
    SEN_TYPE = 2

    def __init__(self, dset_name, data_dir,
                 max_t_len, max_v_len, max_n_len, K, mode="train", sample_mode='uniform',
                 word2idx_path=None, soft_label_dir=None):

        self.dset_name = dset_name
        self.data_dir = data_dir
        self.mode = mode
        self.sample_mode = sample_mode
        self.K = K
        assert K >= 1

        meta_dir = os.path.join(self.data_dir, 'data')

        if (word2idx_path is None) or (not os.path.exists(word2idx_path)):
            logging.info('[WARNING] word2idx_path load failed, use default path.')
            word2idx_path = os.path.join(data_dir, 'vocab_feature', 'word2idx.json')

        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        self.video_feature_dir = os.path.join(self.data_dir, 'video_feature')
        self.duration_file = os.path.join(meta_dir, 'var_video_duration_v1.0.csv')
        self.frame_to_second = self._load_duration()

        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len
        self.max_n_len = max_n_len

        # --- 核心修改点 (1): 只保存路径，不打开文件 ---
        self.soft_label_dir =None
        self.soft_label_index = None
        self.worker_file_handle = None  # 每个worker进程独立管理句柄
        self.soft_label_pkl_path = None

        if mode == 'train' and soft_label_dir is not None:
            index_file = os.path.join(soft_label_dir, 'soft_labels_index.json')
            pkl_file = os.path.join(soft_label_dir, 'soft_labels.pkl')

            if os.path.exists(index_file) and os.path.exists(pkl_file):
                logging.info(f"Loading soft label index from {index_file}")
                self.soft_label_index = load_json(index_file)
                self.soft_label_pkl_path = pkl_file
                self.soft_label_dir = soft_label_dir
            else:
                logging.warning(f"Soft label index/PKL file not found in {soft_label_dir}. Proceeding without distillation.")

        # Load Data
        data_path = os.path.join(self.data_dir, 'data', "var_{}_v1.0.json".format(mode))
        self.data = self._load_data(data_path, mode)
        self.fix_missing()

        # Construct Data List
        self.data_list = []
        for example in self.data:
            example_meta = []
            if 'meta' in example:
                meta = example.pop('meta')
                example_meta.append(meta)
            self.data_list.append((example, example_meta))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        '''
        # 检查当前worker是否已打开文件句柄，如果没有则打开
        if self.soft_label_pkl_path and self.worker_file_handle is None:
            self.worker_file_handle = open(self.soft_label_pkl_path, 'rb')
        '''
        example, example_meta = self.data_list[index]
    
        features, example_meta = self.convert_example_to_features(
            example, example_meta, self.max_t_len, self.max_v_len, self.max_n_len)


        teacher_logits = None

        if self.soft_label_index and self.soft_label_pkl_path and self.mode == 'train':
            # 延迟打开文件句柄 (Worker兼容性)
            if self.worker_file_handle is None:
                try:
                    self.worker_file_handle = open(self.soft_label_pkl_path, 'rb')
                except Exception as e:
                    logging.error(f"Could not open pkl file {self.soft_label_pkl_path}: {e}")
                    self.worker_file_handle = None

            if self.worker_file_handle:
                try:
                    # 3. 还原变量名：从 example['name'] 获取
                    example_name = example.get('name', None)
                    # 备选：如果 example 中没有 name，尝试从 meta 中获取
                    if not example_name and len(example_meta) > 0:
                        example_name = example_meta[0].get('name')

                    if example_name and example_name in self.soft_label_index:
                        index_info = self.soft_label_index[example_name]
                        offset = index_info['offset']
                        length = index_info['len']

                        # 1. 使用 worker 自己的文件句柄读取数据
                        self.worker_file_handle.seek(offset) 
                        serialized_data = self.worker_file_handle.read(length)
                        raw_data = pickle.loads(serialized_data) # List[numpy.ndarray]

                        # [修正 1] 计算足够大的序列长度来容纳所有解码器的输出
                        V = raw_data[0].shape[-1] if len(raw_data) > 0 else self.vocab_size
                        # 3. [关键修改] 计算足够大的序列长度来容纳所有解码器的输出
                        # 不同的解码器输出长度不同 (Pass 1约72, Pass 2约84)，取理论最大值确保不越界
                        max_seq_len = self.max_v_len + self.max_n_len + self.max_t_len

                        padded_logits = torch.full(
                            (self.max_n_len, self.K, max_seq_len, V),
                            fill_value=-1e18,
                            dtype=torch.float32
                        )

                        # 获取当前样本的实际事件数
                        num_events = raw_data[0].shape[0]
                        copy_N = min(num_events, self.max_n_len)

                        # 遍历 K 个解码器层
                        for k in range(len(raw_data)):
                            if k >= self.K: break

                            layer_logits = raw_data[k]
                            if isinstance(layer_logits, np.ndarray):
                                layer_logits = torch.from_numpy(layer_logits)

                                # 获取当前层的实际序列长度
                                cur_L = layer_logits.shape[1]
                                # 截断到容器允许的最大长度
                                copy_L = min(cur_L, max_seq_len)

                                # 将完整的序列数据复制到容器中
                                # 注意：不再是 0:1，而是 :copy_L
                                padded_logits[:copy_N, k, :copy_L, :] = layer_logits[:copy_N, :copy_L, :]

                        teacher_logits = padded_logits

                    else:
                        logging.warning(f"Soft labels for example '{example_name}' not found in index.")
                        teacher_logits = None
                
                except Exception as e:
                    logger.error(f"Error loading soft label: {e}")
                    teacher_logits = None

        # Fallback 逻辑：如果需要蒸馏但加载失败，返回全 -1e18 的张量
        if teacher_logits is None and self.soft_label_dir and self.mode == 'train':
            V = self.vocab_size
            max_seq_len = self.max_v_len + self.max_n_len + self.max_t_len
            teacher_logits = torch.full(
                (self.max_n_len, self.K, max_seq_len, V),
                fill_value=-1e18,
                dtype=torch.float32
            )
           
        # 4. 组装返回值 (保持原版结构)
        final_ret = features
        # 如果 features 不是字典 (虽然在 convert_example_to_features 中它似乎返回的是 dict)，做个兼容
        if not isinstance(final_ret, dict):
            final_ret = {'features': features}

        # 将 teacher_logits 加入返回字典
        if teacher_logits is not None:
            final_ret['teacher_logits'] = teacher_logits

        return final_ret, example_meta       

    def _get_zero_logits(self):
        """辅助函数：创建一个零填充的logits numpy数组，用于异常处理"""
        # 确定序列长度
        if self.K > 1:
            seq_len = self.max_v_len + self.max_n_len + self.max_t_len
        else:
            seq_len = self.max_v_len + self.max_t_len

        return np.zeros((self.max_n_len, seq_len, self.vocab_size), dtype=np.float32)
    
    def set_data_mode(self, mode):
        """mode: `train` or `val` or `test`"""
        assert mode in ['train', 'val', 'test']
        logging.info("Mode {}".format(mode))
        data_path = os.path.join(self.data_dir, 'data', "var_{}_v1.0.json".format(mode))
        self._load_data(data_path, mode)

    def _load_data(self, data_path, mode):
        logging.info("Loading data from {}".format(data_path))
        raw_data = load_json(data_path)
        data = []
        for k, line in raw_data.items():
            if line['split'] != mode: continue
            _line = {}
            valid_n = min(len(line["events"]), self.max_n_len)

            _line["name"] = k
            _line["video_ids"] = [e['video_id'] for e in line['events']]
            _line["events"] = line["events"][:valid_n]
            _line["hypothesis"] = line["hypothesis"]

            _line["video_ids"] = list(set(_line["video_ids"]))
            data.append(_line.copy())
        logging.info("Loading complete! {} examples in total.".format(len(data)))
        return data

    def fix_missing(self):
        """filter our videos with no feature file"""
        missing_video_names = []
        missing_examples = []
        for e in self.data:
            for video_name in e['video_ids']:
                cur_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
                cur_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
                for p in [cur_path_bn, cur_path_resnet]:
                    if not os.path.exists(p):
                        missing_video_names.append(video_name)
                        missing_examples.append(e['name'])

        missing_video_names = list(set(missing_video_names))
        missing_examples = list(set(missing_examples))
        print(f"DEBUG: Total missing examples: {len(missing_examples)}. Remaining data size: {len(self.data)}")
        if len(missing_examples) > 0 or len(missing_video_names) > 0:
            logging.info("Missing {} features (clips/sentences) from {} videos".format(
                len(missing_video_names), len(set(missing_video_names))))
            self.data = [e for e in self.data if e["name"] not in missing_examples]
            print('length: ',len(self.data))

    def _load_duration(self):
        """Load video duration file."""
        frame_to_second = {}
        sampling_sec = 0.5
        with open(self.duration_file, "r") as f:
            for line in f:
                vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                frame_to_second[vid_name] = float(vid_dur) * math.ceil(
                    float(vid_frame) * 1. / float(vid_dur) * sampling_sec) * 1. / float(vid_frame)
        return frame_to_second

    def convert_example_to_features(self, example, example_meta, max_t_len, max_v_len, max_n_len):
        example_name = example["name"]
        video_names = example['video_ids']
        video_features = {}
        for video_name in video_names:
            try:
                feat_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
                feat_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))

                if not os.path.exists(feat_path_resnet):
                    print(f"[CRITICAL ERROR] ResNet Feature NOT FOUND: {feat_path_resnet}")
                if not os.path.exists(feat_path_bn):
                    print(f"[CRITICAL ERROR] BN Feature NOT FOUND: {feat_path_bn}")

                resnet_feat = np.load(feat_path_resnet)
                bn_feat = np.load(feat_path_bn)  # (L, 1024)
                video_features[video_name] = np.concatenate([resnet_feat, bn_feat], axis=1)

            except FileNotFoundError:
                    logging.error(
                    f"Feature file not found for video: {video_name}. This should have been caught by fix_missing().")
                    continue  # Skip if missing, though fix_missing should prevent this

        num_sen = len(example["events"])
        single_video_features = []
        single_video_metas = []
        for clip_idx in range(num_sen):
            cur_data, cur_meta = self.clip_sentence_to_feature(example_name, example["events"][clip_idx],
                                                               video_features)
            single_video_features.append(cur_data)
            single_video_metas.append(cur_meta)

        _mask_idx = example['hypothesis']
        assert _mask_idx < num_sen

        _, f_video_all_mask, f_feat, f_video_temporal_tokens = self._construct_entire_video_features(
            single_video_features)

        # Student model: ALWAYS MASK the hypothesis
        single_video_features[_mask_idx]['video_feature'] = np.zeros_like(
            single_video_features[_mask_idx]['video_feature'])
        single_video_features[_mask_idx]['video_mask'] = [1] * len(single_video_features[_mask_idx]['video_mask'])
        single_video_features[_mask_idx]['video_tokens'] = [self.CLS_TOKEN] + [self.VID_TOKEN] * (
                self.max_v_len - 2) + [self.SEP_TOKEN]
        single_video_metas[_mask_idx]['is_hypothesis'] = True

        _, video_all_mask, feat, video_temporal_tokens = self._construct_entire_video_features(single_video_features)

        input_labels_list = [[] for _ in range(self.K)]
        token_type_ids_list = [[] for _ in range(self.K)]
        input_mask_list = [[] for _ in range(self.K)]
        input_ids_list = [[] for _ in range(self.K)]

        def _fill_data(_idx):
            video_tokens = single_video_features[_idx]['video_tokens']
            video_mask = single_video_features[_idx]['video_mask']
            text_tokens = single_video_features[_idx]['text_tokens']
            text_mask = single_video_features[_idx]['text_mask']
            return video_tokens, video_mask, text_tokens, text_mask

        for _idx in range(self.max_n_len):
            video_tokens, video_mask, text_tokens, text_mask = _fill_data(_idx if _idx < num_sen else 0)

            # * prepare for dec1
            _input_tokens = video_tokens + text_tokens
            _input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in _input_tokens]
            _input_mask = video_mask + text_mask
            _token_type_ids = [self.VID_TYPE] * self.max_v_len + [self.TEX_TYPE] * self.max_t_len

            input_ids_list[0].append(np.array(_input_ids).astype(np.int64))
            token_type_ids_list[0].append(np.array(_token_type_ids).astype(np.int64))
            input_mask_list[0].append(np.array(_input_mask).astype(np.float32))

            _input_labels = \
                [self.IGNORE] * len(video_tokens) + \
                [self.IGNORE if m == 0 else tid for tid, m in zip(_input_ids[-len(text_mask):], text_mask)][1:] + \
                [self.IGNORE]
            input_labels_list[0].append(np.array(_input_labels).astype(np.int64))

            # * prepare for dec2+
            for k_idx in range(1, self.K):
                sen_tokens = [self.SEN_TOKEN] * self.max_n_len
                _input_tokens = video_tokens + sen_tokens + text_tokens
                _input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in _input_tokens]

                sen_mask = [1] * num_sen + [0] * (self.max_n_len - num_sen)
                _input_mask = video_mask + sen_mask + text_mask
                _token_type_ids = [self.VID_TYPE] * self.max_v_len + [self.SEN_TYPE] * self.max_n_len + [
                    self.TEX_TYPE] * self.max_t_len

                input_ids_list[k_idx].append(np.array(_input_ids).astype(np.int64))
                token_type_ids_list[k_idx].append(np.array(_token_type_ids).astype(np.int64))
                input_mask_list[k_idx].append(np.array(_input_mask).astype(np.float32))

                _input_labels = \
                    [self.IGNORE] * (len(video_tokens) + len(sen_tokens)) + \
                    [self.IGNORE if m == 0 else tid for tid, m in zip(_input_ids[-len(text_mask):], text_mask)][1:] + \
                    [self.IGNORE]
                input_labels_list[k_idx].append(np.array(_input_labels).astype(np.int64))

        for k_idx in range(self.K):
            for n_idx in range(num_sen, self.max_n_len):
                input_labels_list[k_idx][n_idx][:] = self.IGNORE
                
        data = dict(
            example_name=example_name,
            num_sen=num_sen,
            encoder_input=dict(
                video_features=feat.astype(np.float32),
                temporal_tokens=np.array(video_temporal_tokens).astype(np.int64),
                video_mask=np.array(video_all_mask).astype(np.float32),
            ),
            unmasked_encoder_input=dict(
                video_features=f_feat.astype(np.float32),
                temporal_tokens=np.array(f_video_temporal_tokens).astype(np.int64),
                video_mask=np.array(f_video_all_mask).astype(np.float32),
            ),
            decoder_input=dict(
                input_ids=input_ids_list,
                input_mask=input_mask_list,
                token_type_ids=token_type_ids_list
            ),
            gt=input_labels_list,
        )
        return data, single_video_metas

    def _construct_entire_video_features(self, single_video_features):
        video_tokens, video_mask, feats, video_temporal_tokens = [], [], [], []
        for idx, clip_feat in enumerate(single_video_features):
            video_tokens += clip_feat['video_tokens'].copy()
            video_mask += clip_feat['video_mask'].copy()
            feats.append(clip_feat['video_feature'].copy())

        if len(single_video_features) < self.max_n_len:
            pad_v_n = self.max_n_len - len(single_video_features)
            video_tokens += [self.PAD_TOKEN] * self.max_v_len * pad_v_n
            video_mask += [0] * self.max_v_len * pad_v_n
            _feat = [np.zeros_like(single_video_features[0]['video_feature'])] * pad_v_n
            feats.extend(_feat)

        for idx in range(self.max_n_len):
            video_temporal_tokens += [idx] * self.max_v_len

        feat = np.concatenate(feats, axis=0)
        return video_tokens, video_mask, feat, video_temporal_tokens

    def clip_sentence_to_feature(self, name, event, video_features):
        event['name'] = name
        event['example_name'] = name
        event['is_hypothesis'] = False
        video_name = event['video_id']
        timestamp = event['timestamp']
        sentence = event['sentence']
        frm2sec = self.frame_to_second[video_name]

        feat, video_tokens, video_mask = self._load_indexed_video_feature(video_features[video_name], timestamp,
                                                                          frm2sec)
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)

        data = dict(
            video_tokens=video_tokens, text_tokens=text_tokens,
            video_mask=video_mask, text_mask=text_mask,
            video_feature=feat.astype(np.float32)
        )

        meta = event

        return data, meta

    @classmethod
    def _convert_to_feat_index_st_ed(cls, feat_len, timestamp, frm2sec):
        st = int(math.floor(timestamp[0] / frm2sec))
        ed = int(math.ceil(timestamp[1] / frm2sec))
        ed = min(ed, feat_len - 1)
        st = min(st, ed - 1)
        assert st <= ed and ed < feat_len, f"st {st} <= ed {ed} < feat_len {feat_len} is False for timestamp {timestamp}"
        return st, ed

    def _load_indexed_video_feature(self, raw_feat, timestamp, frm2sec):
        max_v_l = self.max_v_len - 2
        feat_len = len(raw_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, frm2sec)
        indexed_feat_len = ed - st + 1

        feat = np.zeros((self.max_v_len, raw_feat.shape[1]))

        if indexed_feat_len > max_v_l:
            downsamlp_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(int).tolist()
            assert max(downsamlp_indices) < feat_len
            feat[1:max_v_l + 1] = raw_feat[downsamlp_indices]

            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * max_v_l + [self.SEP_TOKEN]
            mask = [1] * (max_v_l + 2)
        else:
            valid_l = ed - st + 1
            feat[1:valid_l + 1] = raw_feat[st:ed + 1]
            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_l + \
                           [self.SEP_TOKEN] + [self.PAD_TOKEN] * (max_v_l - valid_l)
            mask = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l)

        return feat, video_tokens, mask

    def _tokenize_pad_sentence(self, sentence):
        max_t_len = self.max_t_len
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def convert_ids_to_sentence(self, ids, rm_padding=True, return_sentence_only=True):
        '''
        raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)
        '''
        # 兼容 Tensor
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]

        if not raw_words:
            return ""

        if return_sentence_only:
            words = []
            # [核心修复] 只有当第一个词是 BOS 时才跳过，而不是无脑跳过
            start_idx = 0
            if raw_words[0] == self.BOS_TOKEN:
                start_idx = 1

            for w in raw_words[start_idx:]:
                if w == self.EOS_TOKEN:
                    break
                words.append(w)

            # [核心修复] 防止空串
            if not words:
                return "."
            return " ".join(words)
        else:
            return " ".join(raw_words)

def prepare_batch_inputs(batch, device, non_blocking=False):
    def _recursive_to_device(v):
        if isinstance(v[0], list):
            return [_recursive_to_device(_v) for _v in v]
        elif isinstance(v[0], torch.Tensor):
            return [_v.to(device, non_blocking=non_blocking) for _v in v]
        else: return v

    batch_inputs = dict()

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device, non_blocking=non_blocking)
        elif isinstance(v, list):
            batch_inputs[k] = _recursive_to_device(v)
        elif isinstance(v, dict):
            batch_inputs[k] = prepare_batch_inputs(v, device, non_blocking)

    return batch_inputs


def sentences_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    batch_meta = [[{"name": e['example_name'],
                   "clip_idx": e['clip_idx'],
                   "gt_sentence": e["sentence"],
                   "is_hypothesis": e["is_hypothesis"],
                   } for e in _batch[1]] for _batch in batch]  # change key

    padded_batch = default_collate([e[0] for e in batch])
    return padded_batch, batch_meta

#这是一个训练的聘雇辅助参数，不太建议放到这里
def cal_performance(pred, gold):
    gold = gold[:, -pred.shape[1]:]
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    valid_label_mask = gold.ne(VARDataset.IGNORE)
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(valid_label_mask).sum()

    return n_correct