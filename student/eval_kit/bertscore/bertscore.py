from bert_score import score
import re
import math

class BertScore:
    def __init__(self):
        self._hypo_for_image = {}
        self.ref_for_image = {}
    '''
    def _extract_strings(self, data_list):
        """
        辅助函数：确保列表中的每个元素都是字符串。
        如果元素是字典，尝试提取 'sentence' 或 'caption' 字段。
        """
        result = []
        for item in data_list:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # 尝试获取常见的文本键名
                # VAR 数据集通常使用 'sentence'
                text = item.get('sentence', item.get('caption', None))
                if text is None:
                    # 兜底策略：如果找不到特定键，且字典非空，取第一个值
                    # 或者是空字符串
                    text = list(item.values())[0] if len(item) > 0 else ""
                result.append(str(text))
            else:
                # 其他类型强转为字符串
                result.append(str(item))
        return result   
    '''
    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        vidIds = gts.keys()

        hyp_input = []
        ref_input = []
        same_indices = [] # 用于记录每个视频包含多少个事件
        for id in vidIds:
            # [兼容性修复] 源代码使用 .values()，说明输入是字典 {clip_id: [sent]}
            # 我们保留 sum(values, []) 的逻辑来展平列表
            if isinstance(res[id], dict):
                hypo = sum(res[id].values(), [])
            else:
                hypo = res[id] # 兼容已经展平的情况
            
            if isinstance(gts[id], dict):
                ref = sum(gts[id].values(), [])
            else:
                ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(type(ref) is list)
            assert(len(hypo) == len(ref))
            # 注意：在某些情况下长度可能不一致，如果遇到AssertionError可以注释掉上面这行 

            hyp_input += hypo
            ref_input += ref
            # * average over event - > video
            same_indices.append(len(ref_input))

            # 防止空输入导致 bert_score 报错
        if len(hyp_input) == 0:
            print("[Warning] No valid inputs for BertScore computation.")
            return 0.0
        
        try:
            p, r, f_scores = score(hyp_input, ref_input, \
                model_type='./eval_kit/roberta_large_619fd8c/', num_layers=17, baseline_path='./eval_kit/roberta_large_619fd8c/roberta-large.tsv',\
                lang='en', rescale_with_baseline=True, verbose=True)
        except Exception as e:
            print(f"[Error] BertScore calculation failed: {e}")
            return 0.0
        
        prev_idx = 0 
        aggreg_f1_scores = [] 
        for idx in same_indices: 
            # idx 是累积索引，所以切片是 [prev_idx : idx]
            segment = f_scores[prev_idx: idx]
            if len(segment) > 0:
                aggreg_f1_scores.append(segment.mean().cpu().item())
            else:
                aggreg_f1_scores.append(0.0)
            prev_idx = idx

        # * ignore missing sentence with garbage 
        _sum = sum([x for x in aggreg_f1_scores if not math.isnan(x)])
        return _sum/len(aggreg_f1_scores), aggreg_f1_scores

    def method(self):
        return "BertScore" 