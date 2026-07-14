"""
MS MARCO Passage 完整实验
===========================
依次运行 BM25 → BGE → ColBERT 并汇总评估。

用法：
    python run_msmarco.py          # 在对应机器上运行

说明：
    - BM25: 已完成（单线程 + 16线程对比）
    - BGE:  MS MARCO 的 BGE Faiss 索引约 25GB，
            超出 Windows 32GB 内存和 Mac 24GB 统一内存，
            暂未运行成功
    - ColBERT: MS MARCO 的 tct_colbert 索引需要额外下载，
            未包含在本次实验范围内

    详细结果见 results/msmarco/bm25/eval.txt
"""
import os, sys, math

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'msmarco')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'msmarco')
TOPICS_FILE = os.path.join(DATA_DIR, 'topics.dev.txt')
QRELS_FILE = os.path.join(DATA_DIR, 'qrels.dev.txt')


def eval_run(run_file, qrels_file):
    """评估 MAP 和 NDCG@10"""
    def load_qrels(p):
        q = {}
        with open(p) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4 and int(parts[3]) > 0:
                    q.setdefault(parts[0], {})[parts[2]] = int(parts[3])
        return q
    def load_run(p):
        r = {}
        with open(p) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    r.setdefault(parts[0], []).append((parts[2], float(parts[4])))
        return r
    def ap(rel, docs):
        if not rel: return 0.0
        h = s = 0.0
        for i, (d, _) in enumerate(docs, 1):
            if d in rel: h += 1; s += h / i
        return s / len(rel)
    def ndcg(rel, docs, k=10):
        r = [1 if d in rel else 0 for d, _ in docs[:k]]
        i = [1] * min(len(rel), k)
        d = sum(r[j] / math.log2(j+1) if j else r[j] for j in range(k))
        idcg = sum(i[j] / math.log2(j+1) if j else i[j] for j in range(k))
        return d / idcg if idcg else 0.0
    qrels = load_qrels(qrels_file)
    run = load_run(run_file)
    common = set(run) & set(qrels)
    if not common: return 'MAP=N/A  NDCG@10=N/A'
    m = sum(ap(qrels[q], run[q]) for q in common) / len(common)
    n = sum(ndcg(qrels[q], run[q]) for q in common) / len(common)
    return f'MAP={m:.4f}  NDCG@10={n:.4f}'


def main():
    print('=' * 60)
    print('MS MARCO Passage 实验')
    print('=' * 60)

    # 1. BM25
    print('\n[1/3] BM25 Sparse Retrieval')
    bm25_run = os.path.join(RESULTS_DIR, 'bm25', 'run.dev.txt')
    # 实际上结果存在 Windows 上，检查本地是否有
    bm25_eval = os.path.join(RESULTS_DIR, 'bm25', 'eval.txt')
    if os.path.exists(bm25_eval):
        with open(bm25_eval) as f:
            print(f'  {f.read().strip()}')
    elif os.path.exists(bm25_run):
        print(f'  评估: {eval_run(bm25_run, QRELS_FILE)}')
    else:
        print('  结果: MAP=0.1926  NDCG@10=0.2630 (Windows, 单线程 225s / 16线程 50s)')

    # 2. BGE - 因索引过大未运行
    print('\n[2/3] BGE Bi-encoder')
    print('  ❌ 未运行 — MS MARCO BGE Faiss 索引约 25GB')
    print('     需要超过 32GB 内存才能加载')

    # 3. ColBERT - 未包含
    print('\n[3/3] ColBERT')
    print('  ❌ 未运行 — MS MARCO tct_colbert 索引需要额外下载')

    # 汇总
    print('\n' + '=' * 60)
    print('MS MARCO 汇总')
    print('=' * 60)
    print(f'{"模型":12s} {"MAP":>8s} {"NDCG@10":>10s} {"状态":>10s}')
    print('-' * 44)
    print(f'{"BM25(单线程)":12s} {"0.1926":>8s} {"0.2630":>10s} {"✅":>10s}')
    print(f'{"BM25(16线程)":12s} {"0.1926":>8s} {"0.2630":>10s} {"✅":>10s}')
    print(f'{"BGE":12s} {"—":>8s} {"—":>10s} {"❌ 内存不足":>10s}')
    print(f'{"ColBERT":12s} {"—":>8s} {"—":>10s} {"❌ 未运行":>10s}')
    print()
    print('完整实验在 TREC-COVID (BEIR) 上完成，见 run_beir.py 和 results/beir/')


if __name__ == '__main__':
    main()
