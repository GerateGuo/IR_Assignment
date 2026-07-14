"""
BGE Bi-encoder 稠密检索模型 — 通用实现
========================================
使用 BGE-base-en-v1.5 模型 + Faiss 索引进行稠密检索。

适用于 MS MARCO 和 BEIR（TREC-COVID 等）数据集。

用法示例：
    from models.bge import run_bge_search
    
    run_bge_search(
        index_name='msmarco-v1-passage.bge-base-en-v1.5',
        topics_file='../data/msmarco/topics.dev.txt',
        output_file='../results/msmarco/bge/run.dev.txt'
    )

原理说明：
    Bi-encoder 将查询和文档分别编码为向量（embedding），
    然后用余弦相似度或内积计算两者的匹配得分。
    
    - 文档编码：预计算后存入 Faiss 索引（已下载）
    - 查询编码：运行时用 BGE 模型实时编码
    - 搜索：在 Faiss 索引中找 top-K 最相似的文档向量

与 BM25 的区别：
    - BM25：基于关键词精确匹配（稀疏向量）
    - BGE：基于语义相似度（稠密向量），能处理同义词/近义词
    - BGE 通常比 BM25 效果更好，但需要 GPU 加速

⚠️ 多线程说明：
    BGE 与 BM25 不同，**不适合** Python 层多线程：
    - BGE 的编码器（PyTorch 模型）运行在 GPU 上，多线程同时调用会导致 CUDA 错误
    - Faiss 搜索内部已通过 OpenMP 自动并行（C++ 层多线程）
    - GPU 编码本身已经是大规模并行计算
    ✅ 正确优化方式：用 `batch_size` 批量编码查询（而非多线程）
    ✅ Pyserini CLI 支持 --threads 和 --batch-size 参数
"""

import os
import sys
import time


def setup_java_env():
    """配置 JAVA_HOME 和 PATH。"""
    java_home = r'C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot'
    os.environ.setdefault('JAVA_HOME', java_home)
    jvm_path = os.path.join(java_home, r'bin\server')
    if jvm_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = jvm_path + ';' + os.environ.get('PATH', '')


def load_topics(filepath):
    """加载查询文件。"""
    topics = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                topics[parts[0]] = parts[1]
    return topics


def run_bge_search(index_name, topics_file, output_file,
                   encoder_model='BAAI/bge-base-en-v1.5',
                   query_prefix='Represent this sentence for searching relevant passages:',
                   hits=1000, batch_size=512):
    """
    使用 BGE Bi-encoder 模型执行稠密检索。
    
    参数:
        index_name    : Faiss 预建索引名称
        topics_file   : 查询文件路径
        output_file   : 输出结果路径（TREC 格式）
        encoder_model : HuggingFace 上的 BGE 模型名称
        query_prefix  : BGE 模型要求的查询前缀
        hits          : 每个查询返回的文档数
        batch_size    : 批量编码大小（默认 512，充分利用 GPU 并行）
    
    ⚠️ 多线程说明：
        BGE 不适合 Python 多线程（GPU 编码不共享），
        正确加速方式是增大 batch_size 让 GPU 一次编码更多查询。
        Faiss 搜索内部已通过 OpenMP 自动多线程。
        
    也可直接用 Pyserini CLI 替代：
        python -m pyserini.search.faiss ^
          --threads 16 --batch-size 512 ^
          --encoder-class auto --encoder BAAI/bge-base-en-v1.5 ^
          --query-prefix "Represent this sentence for searching relevant passages:" ^
          --index msmarco-v1-passage.bge-base-en-v1.5 ^
          --topics msmarco-v1-passage.dev ^
          --output run.bge.dev.txt
    """
    setup_java_env()
    
    from pyserini.search.faiss import FaissSearcher
    from pyserini.encode import AutoQueryEncoder
    
    print(f'Loading index: {index_name}')
    print(f'Loading encoder: {encoder_model}')
    
    # 初始化查询编码器（BGE 在 GPU 上跑更快）
    # 如果遇到 OOM，可以加 device='cpu' 参数
    encoder = AutoQueryEncoder(
        model_name=encoder_model,
        query_prefix=query_prefix,
        device='cuda:0'  # 使用 GPU（RTX 5070 Ti）
    )
    
    searcher = FaissSearcher.from_prebuilt_index(index_name, encoder)
    
    print(f'Loading topics: {topics_file}')
    topics = load_topics(topics_file)
    print(f'  Loaded {len(topics)} queries')
    
    print('Running BGE search...')
    t0 = time.time()
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (qid, query) in enumerate(topics.items()):
            hits_list = searcher.search(query, k=hits)
            for rank, hit in enumerate(hits_list):
                f.write(f'{qid} Q0 {hit.docid} {rank+1} {hit.score:.6f} BGE\n')
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f'  {i+1}/{len(topics)} queries ({elapsed:.0f}s)')
    
    t = time.time() - t0
    print(f'Done! {len(topics)} queries in {t:.1f}s')
    print(f'Output: {output_file}')


def run_bge_msmarco():
    """在 MS MARCO 上跑 BGE。"""
    run_bge_search(
        index_name='msmarco-v1-passage.bge-base-en-v1.5',
        topics_file='../data/msmarco/topics.dev.txt',
        output_file='../results/msmarco/bge/run.dev.txt'
    )


def run_bge_trec_covid():
    """在 TREC-COVID 上跑 BGE。"""
    run_bge_search(
        index_name='beir-v1.0.0-trec-covid.bge-base-en-v1.5',
        topics_file='../data/beir/trec-covid/topics.txt',  # 需要下载该数据集
        output_file='../results/beir/bge/run.trec-covid.txt'
    )


if __name__ == '__main__':
    run_bge_msmarco()
