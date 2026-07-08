# Reasoner-KD

**Knowledge Distillation for Visual Abductive Reasoning (VAR)**

Reasoner-KD distills a strong *teacher* reasoner into a lightweight *student* model
for the Visual Abductive Reasoning (VAR) task. The full pipeline runs in four stages:

```
① Train Teacher  →  ② Generate Soft Labels  →  ③ Train Student  →  ④ Evaluate
```

The teacher produces soft labels (logit distributions) that are cached to disk, and
the student is then trained under the supervision of both the ground-truth (hard)
labels and the teacher's soft labels.

---

## Table of Contents

- [1. Train the Teacher Model](#1-train-the-teacher-model)
- [2. Generate Soft Labels](#2-generate-soft-labels)
- [3. Train the Student Model](#3-train-the-student-model)
- [4. Evaluation](#4-evaluation)
- [Output File Reference](#output-file-reference)

---

## 1. Train the Teacher Model

**Command format**

```bash
bash scripts/train.sh ${N_GPUS} ${MODEL_NAME} --model_mode teacher
```

**Example (single GPU)**

```bash
bash scripts/train.sh 1 Reasoner --model_mode teacher
```

After training completes, you will obtain, under a timestamped results directory:

- a model weight file — e.g. `results/VAR_Reasoner_2025_09_21_11_44_29/model.chkpt`
- a config file — e.g. `results/VAR_Reasoner_2025_09_21_11_44_29/model.cfg.json`
- 5 evaluation-related files

> **Note:** The teacher's evaluation files are **not** required by the later stages
> and can be ignored.

---

## 2. Generate Soft Labels

Run `generate_soft_labels.py` with the paths to **your own** trained teacher
checkpoint and config.

**Command**

```bash
python src/generate_soft_labels.py \
    --checkpoint_path D:/code/model/my/teacher/results/VAR_Reasoner_2025_11_08_00_02_09/model.chkpt \
    --config_path     D:/code/model/my/teacher/results/VAR_Reasoner_2025_11_08_00_02_09/model.cfg.json \
    --data_dir        D:/code/model/my/data/VAR \
    --output_dir      D:/code/model/my/teacher/soft_labels_pkl \
    --batch_size      32
```

This produces one **`.pkl`** data file and one **`.json`** index file. Soft labels are
written to disk via streaming write + **memory mapping**: all soft labels are stored as
a single large, contiguous NumPy array, and the accompanying JSON file records the index
position of every sample within that array.

**Full parameter list (with GloVe embeddings)**

```bash
python src/generate_soft_labels.py \
    --checkpoint_path "results/YOUR_TEACHER_MODEL_DIR/best.chkpt" \
    --config_path     "results/YOUR_TEACHER_MODEL_DIR/best.cfg.json" \
    --data_dir        "D:/path/to/your/VAR" \
    --glove_path      "D:/path/to/glove" \
    --glove_version   "vocab_glove.6B.300d.pt" \
    --output_dir      "D:/code/model/my/teacher/soft_labels_pkl" \
    --batch_size      16
```

**(Optional) Evaluate the teacher**

```bash
python eval_kit/evaluate_models.py results/VAR_Reasoner_2025_12_23_00_18_54/model_best_greedy_pred_test.json
```

---

## 3. Train the Student Model

```bash
bash scripts/train.sh 1 Reasoner_Student
```

> **Design note — where distillation is applied:**
> `teacher_logits` are used to guide the **final output** only; they should **not** be
> applied to the intermediate steps of the cascade. Each cascade step's loss is computed
> from the **hard labels** alone. Knowledge distillation is introduced **only after the
> cascade loop finishes**, when the final loss is computed.

After training completes, you will obtain a model weight file
(e.g. `results/VAR_Reasoner_Student_2025_09_21_11_44_29/model.chkpt`) and a config file
(e.g. `results/VAR_Reasoner_Student_2025_09_21_11_44_29/model.cfg.json`).

---

## 4. Evaluation

```bash
python eval_kit/evaluate_models.py results/VAR_Reasoner_Student_2025_09_21_11_44_29/model_best_greedy_pred_test.json
```

**Example output**

```
[Separate Observed]   METEOR 25.30  Bleu@4 4.16  CIDEr 33.04  ROUGE_L 23.55  BERT_S 28.92
[Separate Hypothesis] METEOR 24.14  Bleu@4 4.57  CIDEr 34.07  ROUGE_L 23.54  BERT_S 29.83
```

---

## Output File Reference

During training and evaluation, several prediction and metric files are produced.
Their meanings are as follows:

| File | Meaning | Description |
|------|---------|-------------|
| `model_tmp_greedy_pred_test_0.json` | **Temporary prediction file** | Prediction results from a **single GPU process** (`_0` = rank 0). Each entry corresponds to one test sample and contains the generated sentence (`sentence`), the ground-truth (`gt_sentence`), etc. In multi-GPU training every process writes one such file (`_0.json`, `_1.json`, …); useful for debugging a single process's output. |
| `model_tmp_greedy_pred_test.json` | **Merged temporary prediction file** | The main process merges all per-GPU temporary files into this one. In single-GPU training it is essentially a copy of `_0.json`. This complete-test-set file is the **input to the evaluation script**. |
| `model_tmp_greedy_pred_test_all_metrics.json` | **Temporary metrics file** | All metric scores (BLEU, METEOR, CIDEr, …) computed from the merged temporary prediction file. Records the results of a **non-best** model (i.e. the model at the end of an epoch, which is not necessarily the highest-CIDEr checkpoint). |
| `model_best_greedy_pred_test.json` | **Best model's prediction file** | When an epoch's CIDEr score **exceeds** all previous epochs, the current weights are saved (`*.chkpt`) and the temporary prediction file is **renamed** to `*_best_*_test.json`. Stores the actual outputs of the best-performing model on the test set — useful for qualitative analysis and case studies. |
| `model_best_greedy_pred_test_all_metrics.json` | **Best model's metrics file** | **The key result file.** Records the metric scores corresponding to the best model's predictions. The **final performance** of your model for this run is reported here — compare its metrics (especially CIDEr) against the original paper and baselines. |

---

### Metric legend

- **BLEU@n / METEOR / ROUGE_L** — n-gram overlap and alignment-based text-generation metrics.
- **CIDEr** — consensus-based caption evaluation; the primary metric for model selection here.
- **BERT_S** — BERTScore, a contextual-embedding similarity metric.
