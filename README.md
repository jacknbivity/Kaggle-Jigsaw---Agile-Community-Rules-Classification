# Kaggle Jigsaw — Agile Community Rules Classification

[![Competition](https://img.shields.io/badge/Kaggle-Jigsaw%20Agile%20Community%20Rules-blue)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview)
[![Private LB](https://img.shields.io/badge/Private%20LB-0.92602-green)](#)
[![Public LB](https://img.shields.io/badge/Public%20LB-0.931-brightgreen)](#)
[![Rank](https://img.shields.io/badge/Rank-22%2F2445%20(Top%200.9%25)-blueviolet)](#)

**判断 Reddit 评论是否违反子版块社区规则** 的二分类 NLP 任务，评估指标 **Column-averaged ROC-AUC**。

---

## 🏆 成绩

| 榜单 | AUC | 排名 |
|------|-----|------|
| Public Leaderboard | **0.931** | — |
| Private Leaderboard | **0.92602** | **🥈 22 / 2445** (Top 0.9%) |

---

## 📁 项目结构

```
├── blendmymodel-ed5bf4.ipynb   # 主 Notebook（数据预处理 + 编码器模型 + 模型融合）
├── Model1/                      # Qwen3-4B + 混合损失（多任务辅助分类）
│   ├── constants.py             # 模型路径 & 超参数
│   ├── utils.py                 # 数据构建 & RAG 检索模块
│   ├── train.py                 # QLoRA + CustomSFTTrainer
│   └── inference.py             # vLLM + RAG + TTA 推理
├── Model2/                      # Qwen3-4B + 混合损失（同 Model1，不同 seed）
├── Model3/                      # Qwen-3-4B + 纯 SFT
├── Model4/                      # Llama-3.2-3B + 纯 SFT
└── README.md
```

---

## 🧠 模型全景

| # | 模型 | 基座 | 训练策略 | CV AUC |
|---|------|------|---------|--------|
| 1 | Qwen3-4B + Aux | `Qwen3-4B-Instruct-2507` | QLoRA + 混合损失 | **0.923** |
| 2 | Qwen3-4B + Aux | `Qwen3-4B-Instruct-2507` | QLoRA + 混合损失 | 0.923 |
| 3 | Qwen3-4B | `Qwen-3/4B` | QLoRA + 纯 SFT | 0.920 |
| 4 | Llama-3.2-3B | `Llama-3.2-3B-Instruct` | QLoRA + 纯 SFT | 0.920 |
| 5 | DeBERTa-v3 | `deberta-v3-base` | 全参 + 多任务 | 0.912 |
| 6 | E5-BERT | `e5-base-v2` | 全参 + 多任务 | 0.911 |
| 7 | BGE-Combined | `bge-small-en-v1.5` | BERT+TextCNN+FastText | 0.912 |

---

## 🔧 核心技术

### 1. 高效训练：QLoRA + DeepSpeed

- **4-bit NF4 量化** 加载基座模型，显存降至 1/4
- **LoRA rank=64**，仅训练 ~3% 参数（~132M / 4B）
- **DeepSpeed ZeRO-2** 分片优化器 + 梯度
- **Gradient Checkpointing** 激活值重算，显存 20GB → **~5GB**
- 在 **T4×2 GPU** 上完成 4B 模型微调

### 2. 多任务辅助损失

```
总损失 = 0.8 × SFT 主损失 + 0.2 × 规则类型分类辅助损失
```

- 主任务：`Violation:` 后生成 Yes/No（Causal LM CE Loss）
- 辅助任务：最后一个 hidden state → `nn.Linear(N+1)` → 识别违规规则类型（CrossEntropy）
- 共享 Transformer backbone，迫使模型学到规则感知表征

### 3. RAG + TTA 推理增强

- **TF-IDF 检索**：同规则下检索 Top-K 高相似度标注样本
- **TTA**：4 种正/负例组合 × 概率平均
- 全 CPU 检索，不占用推理 GPU 显存

### 4. 加权模型融合

7 个异构模型基于规则级 CV AUC 的 softmax 动态权重加权平均。

---

## 🚀 快速开始

### 环境

```bash
pip install trl==0.21.0 peft accelerate datasets bitsandbytes==0.46.1
pip install vllm==0.10.0 logits-processor-zoo==0.2.1
pip install deepspeed==0.17.4
```

### 训练

```bash
# LLM 模型（QLoRA + DeepSpeed）
accelerate launch --config_file accelerate_config.yaml Model1/train.py

# 编码器模型（全参微调）
python train_deberta.py
python train_e5_bert.py
```

### 推理

```bash
python Model1/inference.py   # → Qwen3_4B_7_submission.csv
python Model3/inference.py   # → Qwen3_submission.csv
python Model4/inference.py   # → llama_submission.csv
```

### 融合

运行 Notebook 最后的 ensemble cell，生成 `submission.csv`。

---

## 📊 消融实验

| 配置 | 基线 | +辅助损失 | +RAG TTA | +7 模型融合 |
|------|:----:|:--------:|:--------:|:----------:|
| AUC | 0.918 | +0.004 | +0.002 | +0.007 |
| 累计 | 0.918 | 0.922 | 0.924 | **0.931 / 0.926** |

---

## 📝 竞赛信息

- **竞赛链接**：[Jigsaw - Agile Community Rules Classification](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/overview)
- **任务**：给定 Reddit 评论 + 子版块规则，判断是否违规
- **评估指标**：Column-averaged ROC-AUC
- **难点**：4 类规则不可见、大量俚语/特殊字符、推理时间严格限制

---

## 📄 License

MIT
