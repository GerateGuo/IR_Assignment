"""
BM25 稀疏检索模型 — 多线程版本
===================================
通过 ThreadPoolExecutor 并行搜索多个查询，
能充分利用 CPU 多核心，大幅缩短检索时间。

速度对比（6980 个查询，MS MARCO）：
  单线程: 225 秒
  16 线程: 49.7 秒（快 4.5 倍）

用法示例：
    from models.bm25.threaded import run_bm25_search_threaded, evaluate
    
    run_bm25_search_threaded(
        index_name='msmarco-v1-passage',
        topics_file='../data/msmarco/topics.dev.txt',
        output_file='../results/msmarco/bm25/run.threaded.txt',
        num_threads=16
    )
    
    # 评估
    evaluate(
        run_file='../results/msmarco/bm25/run.threaded.txt',
        qrels_file='../data/msmarco/qrels.dev.txt',
        output_file='../results/msmarco/bm25/eval.txt'
    )
"""

import os
import sys
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# 环境配置（Windows 上需要）
# ============================================================
def setup_java_env():
    """配置 JAVA_HOME 和 PATH，使 pyjnius 能加载 JVM。"""
    java_home = r'C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot'
    os.environ.setdefault('JAVA_HOME', java_home)
    jvm_path = os.path.join(java_home, r'bin\server')
    if jvm_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = jvm_path + ';' + os.environ.get('PATH', '')


# ============================================================
# 数据加载
# ============================================================
def load_topics(filepath):
    """加载查询文件。每行格式：query_id  query_text"""
    topics = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                topics[parts[0]] = parts[1]
    print(f'  Loaded {len(topics)} queries')
    return topics


def load_qrels(filepath):
    """加载相关性判断。格式：qid  0  docid  relevance"""
    qrels = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                if rel > 0:
                    qrels.setdefault(qid, {})[docid] = rel
    print(f'  Loaded {len(qrels)} queries with judgments')
    return qrels


def load_run(filepath):
    """加载检索结果（TREC 格式）。"""
    run = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank, score, _ = parts
                run.setdefault(qid, []).append((docid, float(score), int(rank)))
    print(f'  Loaded {len(run)} queries in run')
    return run


# ============================================================
# 多线程 BM25 检索
# ============================================================
def _search_single(args):
    """单个查询的搜索函数（供多线程内部调用）。"""
    qid, query, searcher, hits_k = args
    results = searcher.search(query, k=hits_k)
    return qid, [(hit.docid, hit.score, rank + 1) for rank, hit in enumerate(results)]


def run_bm25_search_threaded(index_name, topics_file, output_file,
                             k1=0.9, b=0.4, hits=1000, num_threads=16):
    """
    使用 BM25 模型执行多线程检索。
    
    参数:
        index_name  : Pyserini 预建索引名称
        topics_file : 查询文件路径
        output_file : 输出结果路径（TREC 格式）
        k1, b       : BM25 参数
        hits        : 每个查询返回的文档数
        num_threads : 并行线程数（建议 CPU 核心数，如 16）
    """
    setup_java_env()
    from pyserini.search.lucene import LuceneSearcher
    
    # 1. 加载索引
    print(f'Loading index: {index_name}')
    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    searcher.set_bm25(k1, b)
    
    # 2. 加载查询
    print(f'Loading topics: {topics_file}')
    topics = load_topics(topics_file)
    
    # 3. 多线程并行搜索
    print(f'Running BM25 search with {num_threads} threads...')
    query_args = [(qid, query, searcher, hits) for qid, query in topics.items()]
    
    results_dict = {}
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(_search_single, a): a[0] for a in query_args}
        for i, future in enumerate(as_completed(futures)):
            qid, hits_list = future.result()
            results_dict[qid] = hits_list
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(f'  {i+1}/{len(topics)} queries ({elapsed:.0f}s)')
    
    # 4. 写入 TREC 格式结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for qid in topics:
            if qid in results_dict:
                for docid, score, rank in results_dict[qid]:
                    f.write(f'{qid} Q0 {docid} {rank} {score:.6f} BM25\n')
    
    t = time.time() - t0
    print(f'Done! {len(topics)} queries in {t:.1f}s with {num_threads} threads')
    print(f'Output: {output_file}')


# ============================================================
# 评估（与单线程版本共用，方便独立使用）
# ============================================================
def average_precision(relevant_docs, ranked_docs):
    """单个查询的 Average Precision。"""
    if not relevant_docs:
        return 0.0
    hits = 0.0
    sum_prec = 0.0
    for i, (docid, _, _) in enumerate(ranked_docs, 1):
        if docid in relevant_docs:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(relevant_docs)


def dcg(relevances, k):
    """Discounted Cumulative Gain@k。"""
    relevances = relevances[:k]
    val = 0.0
    for i, rel in enumerate(relevances, 1):
        val += rel if i == 1 else rel / math.log2(i)
    return val


def ndcg_at_k(relevant_docs, ranked_docs, k=10):
    """Normalized DCG@k。"""
    relevances = [1 if d in relevant_docs else 0 for d, _, _ in ranked_docs[:k]]
    ideal = [1] * min(len(relevant_docs), k)
    d = dcg(relevances, k)
    idcg = dcg(ideal, k)
    return d / idcg if idcg > 0 else 0.0


def evaluate(run_file, qrels_file, output_file=None):
    """
    评估检索结果，计算 MAP 和 NDCG@10。
    
    返回: (map_score, ndcg10_score)
    """
    print('Loading qrels...')
    qrels = load_qrels(qrels_file)
    
    print('Loading run...')
    run = load_run(run_file)
    
    common = set(run.keys()) & set(qrels.keys())
    print(f'  {len(common)} common queries')
    
    ap_sum = 0.0
    ndcg_sum = 0.0
    for qid in common:
        ap_sum += average_precision(qrels[qid], run[qid])
        ndcg_sum += ndcg_at_k(qrels[qid], run[qid], k=10)
    
    map_score = ap_sum / len(common) if common else 0.0
    ndcg10_score = ndcg_sum / len(common) if common else 0.0
    
    print(f'\n{"=" * 50}')
    print('Evaluation Results')
    print(f'{"=" * 50}')
    print(f'Run     : {run_file}')
    print(f'MAP     = {map_score:.4f}')
    print(f'NDCG@10 = {ndcg10_score:.4f}')
    print(f'Queries = {len(common)}')
    print(f'{"=" * 50}')
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f'Run     : {run_file}\n')
            f.write(f'MAP     = {map_score:.4f}\n')
            f.write(f'NDCG@10 = {ndcg10_score:.4f}\n')
            f.write(f'Queries = {len(common)}\n')
        print(f'Saved to: {output_file}')
    
    return map_score, ndcg10_score
