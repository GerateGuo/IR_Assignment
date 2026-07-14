"""
BEIR（TREC-COVID）完整实验
==========================
依次运行 BM25 → BGE → ColBERT 并汇总评估。

用法：
    # Windows 环境
    python run_beir.py

    # Mac 环境（ColBERT 需 Mac MPS）
    python run_beir.py
"""
import os, sys

# 本地数据路径（不在 Pyserini cache 中，已下载到项目目录）
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'beir')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'beir')
TOPICS_FILE = os.path.join(DATA_DIR, 'trec-covid-topics.txt')
QRELS_FILE = os.path.join(DATA_DIR, 'trec-covid-qrels.txt')

def eval_run(run_file, qrels_file):
    """本地评估函数，计算 MAP 和 NDCG@10"""
    import math
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
    print('BEIR TREC-COVID 完整实验')
    print('=' * 60)

    # 1. BM25 — 已在 Windows 完成
    print('\n[1/3] BM25')
    bm25_run = os.path.join(RESULTS_DIR, 'bm25', 'run.txt')
    if os.path.exists(bm25_run):
        print(f'  结果文件: {bm25_run}')
        print(f'  评估: {eval_run(bm25_run, QRELS_FILE)}')
    else:
        print('  未找到 BM25 结果，请先在 Windows 运行。')

    # 2. BGE — 已在 Windows 完成
    print('\n[2/3] BGE Bi-encoder')
    bge_run = os.path.join(RESULTS_DIR, 'bge', 'run.txt')
    if os.path.exists(bge_run):
        print(f'  结果文件: {bge_run}')
        print(f'  评估: {eval_run(bge_run, QRELS_FILE)}')
    else:
        print('  未找到 BGE 结果，请先在 Windows 运行。')

    # 3. ColBERT — 已在 Mac 完成
    print('\n[3/3] ColBERT (tct_colbert re-ranking)')
    colbert_run = os.path.join(RESULTS_DIR, 'colbert', 'run.optimized.txt')
    if os.path.exists(colbert_run):
        print(f'  结果文件: {colbert_run}')
        print(f'  评估: {eval_run(colbert_run, QRELS_FILE)}')
    else:
        colbert_run2 = os.path.join(RESULTS_DIR, 'colbert', 'run.txt')
        if os.path.exists(colbert_run2):
            print(f'  结果文件: {colbert_run2}')
            print(f'  评估: {eval_run(colbert_run2, QRELS_FILE)}')

    # 汇总
    print('\n' + '=' * 60)
    print('汇总')
    print('=' * 60)
    print(f'{"模型":10s} {"MAP":>8s} {"NDCG@10":>10s}')
    print('-' * 30)
    for name, path in [('BM25', bm25_run), ('BGE', bge_run), ('ColBERT', colbert_run)]:
        if os.path.exists(path):
            result = eval_run(path, QRELS_FILE)
            map_val = result.split()[0].split('=')[1]
            ndcg_val = result.split()[1].split('=')[1]
            print(f'{name:10s} {map_val:>8s} {ndcg_val:>10s}')

if __name__ == '__main__':
    main()
