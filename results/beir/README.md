# TREC-COVID 实验结果对比

## 三种模型对比

| 模型 | MAP | NDCG@10 | 环境 | 耗时 |
|:---|:---:|:-------:|:----|:---:|
| BM25（稀疏检索） | 0.1871 | 0.6695 | CPU | 2.5s |
| BGE（稠密检索） | **0.2581** | **0.8461** | GPU (RTX 5070 Ti) | 1.3s |
| ColBERT 初始版 | 0.0678 | 0.5477 | CPU | 232s (50查询) |
| ColBERT 优化版 | 0.0803 | 0.7601 | MPS (Mac M5) | **28s** |

## ColBERT 优化过程

| 版本 | 池化方式 | L2 归一化 | 硬件 | MAP | NDCG@10 |
|:---|:---------:|:---------:|:----|:---:|:-------:|
| v1 | mean pool | ❌ | CPU | 0.0678 | 0.5477 |
| v2 | [CLS] | ❌ | CPU | 0.0688 | 0.5616 |
| v3 | [CLS] + L2 | ✅ | CPU | — | — |
| **v4 (最优)** | **mean pool + L2** | ✅ | **MPS + float16** | **0.0803** | **0.7601** |

## 关键发现

### 1. 稀疏 vs 稠密
- **BM25（稀疏）**：基于关键词匹配，在医学领域表现稳定
- **BGE（稠密 Bi-encoder）**：语义理解更强，比 BM25 提升 **+38% MAP**
- **ColBERT（稠密 Late interaction）**：token 级匹配，优化后 NDCG@10 大幅提升

### 2. 领域迁移的影响
- tct_colbert-v2 在 **MS MARCO（网页搜索）** 上训练
- TREC-COVID 是 **医学文献** 数据集，领域差异大
- BGE 作为通用 embedding 模型，zero-shot 迁移效果优于 tct_colbert

### 3. 池化策略的重要性
- mean pooling 优于 [CLS] pooling（在 tct_colbert 上）
- **L2 归一化**是关键的优化步骤，NDCG@10 从 0.55 提升到 0.76

### 4. 性能与资源权衡
- GPU（RTX 5070 Ti）：BGE 1.3s 跑完 → 适合稠密检索
- MPS（Mac M5 24GB）：ColBERT 28s → 统一内存无显存限制
- CPU：慢但稳定，适合小规模实验

## 文件说明

```
results/beir/
├── bm25/run.txt          — BM25 搜索结果（50查询 × 1000结果）
├── bge/run.txt           — BGE 搜索结果（50查询 × 1000结果）
└── colbert/
    ├── run.optimized.txt  — ColBERT 优化版结果
    └── run.initial.txt    — ColBERT 初始版结果（未优化）
```

## 评估指标说明

- **MAP（Mean Average Precision）**：平均准确率，衡量所有相关文档的排序质量
- **NDCG@10（Normalized Discounted Cumulative Gain@10）**：前 10 个结果的排序质量，对头部排序更敏感
