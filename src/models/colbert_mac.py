"""
Mac 版 tct_colbert 重排序 — TREC-COVID
=========================================
用 transformers 直接加载模型，datasets 下载文档库
"""
import os, sys, time, gc, numpy as np

# Mac 代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['TRANSFORMERS_OFFLINE'] = '0'  # 可以联网

print('Loading tct_colbert model...')
sys.stdout.flush()

from transformers import AutoTokenizer, AutoModel

# 加载 tct_colbert-v2 模型（双编码器共享同一权重）
model_name = 'castorini/tct_colbert-v2-msmarco'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to('mps')
model.eval()
print(f'Model loaded on MPS (Apple GPU)')
sys.stdout.flush()

def encode(texts, prefix='', batch_size=128):
    """用模型的 [CLS] 向量作为文档/查询表示（MPS 加速 + float16）"""
    import torch
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i:i+batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True,
                          return_tensors='pt', max_length=256)
        inputs = {k: v.to('mps') for k, v in inputs.items()}
        with torch.no_grad():
            with torch.amp.autocast('mps'):  # float16 混合精度
                outputs = model(**inputs)
        # 最优：mean pooling + L2 归一化
        embeds = outputs.last_hidden_state.mean(dim=1).float().cpu().numpy()
        # L2 归一化
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds = embeds / norms
        all_embeds.append(embeds)
        torch.mps.empty_cache()
        gc.collect()
    return np.vstack(all_embeds)

# 加载 TREC-COVID 文档库
print('Loading TREC-COVID corpus...')
from datasets import load_dataset
ds = load_dataset('BeIR/trec-covid', 'corpus', split='corpus')
print(f'Corpus: {len(ds)} documents')
sys.stdout.flush()

# 建立 docid -> text 映射
doc_map = {}
for doc in ds:
    doc_map[doc['_id']] = f"{doc.get('title', '')} {doc.get('text', '')}"
print(f'Doc map: {len(doc_map)} entries')
sys.stdout.flush()

# 读 topics
topics = []
with open('/Users/gerate/Desktop/IR_Assignment/data/beir/trec-covid-topics.txt', encoding='utf-8') as f:
    for line in f:
        p = line.strip().split('\t', 1)
        if len(p) == 2: topics.append((p[0], p[1]))

# 读 BM25 top-100
print('Loading BM25 top-100...')
bm25_top = {}
with open('/Users/gerate/Desktop/IR_Assignment/results/beir/bm25/run.txt') as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 6 and int(p[3]) <= 100:
            bm25_top.setdefault(p[0], []).append(p[2])
sys.stdout.flush()

# 收集所有 unique BM25 top-100 文档并批量预编码
all_doc_ids = set()
for docs in bm25_top.values():
    all_doc_ids.update(docs)
print(f'Unique BM25 top docs: {len(all_doc_ids)}')

print('Encoding all documents (batch MPS + float16)...')
t0 = time.time()
all_texts = [doc_map.get(d, '') for d in all_doc_ids]
all_doc_vecs = encode(all_texts, batch_size=128)
doc_emb_map = dict(zip(all_doc_ids, all_doc_vecs))
print(f'Encoded {len(all_doc_ids)} docs in {time.time()-t0:.0f}s')
sys.stdout.flush()

# 预编码所有 query
print('Encoding queries...')
q_embs = {}
for qid, query in topics:
    q_embs[qid] = encode([query])[0]

# 重排序
print('Re-ranking...')
t0 = time.time()
out = '/Users/gerate/Desktop/IR_Assignment/results/beir/colbert/run.optimized.txt'
with open(out, 'w') as f:
    for i, (qid, query) in enumerate(topics):
        docs = bm25_top.get(qid, [])
        valid = [d for d in docs if d in doc_emb_map]
        if not valid: continue
        
        d_embs = np.array([doc_emb_map[d] for d in valid])
        scores = np.dot(d_embs, q_embs[qid])
        ranked = sorted(zip(valid, scores), key=lambda x: -x[1])
        
        for r, (did, sc) in enumerate(ranked):
            f.write(f'{qid} Q0 {did} {r+1} {sc:.6f} TCT_COLBERT\n')
        
        if (i+1) % 10 == 0:
            print(f'  {i+1}/{len(topics)} ({time.time()-t0:.0f}s)')
            sys.stdout.flush()

print(f'Done! {time.time()-t0:.1f}s')
print(f'Output: {out}')
