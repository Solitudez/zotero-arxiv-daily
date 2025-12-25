import numpy as np
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime

def rerank_paper(candidate:list[ArxivPaper],corpus:list[dict],model:str='Qwen/Qwen3-Embedding-0.6B') -> list[ArxivPaper]:
    """
    使用 Qwen3 Embedding 模型对候选论文进行重排序。
  
    Args:
        candidate: arXiv 候选论文列表（待排序）
        corpus: Zotero 用户收藏的论文列表（用户兴趣偏好）
        model: 嵌入模型名称
  
    Returns:
        按相似度得分排序后的候选论文列表
    """
    # Qwen3-Embedding 需要 transformers>=4.51.0 和 sentence-transformers>=2.7.0
    encoder = SentenceTransformer(model, trust_remote_code=True)
    # 按日期排序 corpus，从新到旧
    corpus = sorted(
        corpus,
        key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
        reverse=True
    )
  
    # 时间衰减权重：越新的论文权重越高
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
  
    # ===== 根据 Qwen3 官方最佳实践编码 =====
    # corpus（用户收藏）作为 document，不使用 prompt
    corpus_feature = encoder.encode(
        [paper['data']['abstractNote'] for paper in corpus],
        batch_size=8,           # 控制内存使用，防止 GitHub Actions OOM
        show_progress_bar=True  # 显示进度便于调试
    )
  
    # candidate（新论文）作为 query，使用 prompt_name="query"
    candidate_feature = encoder.encode(
        [paper.summary for paper in candidate],
        prompt_name="query",    # 关键：Qwen3 query 需要添加 prompt 以获得最佳效果
        batch_size=8,
        show_progress_bar=True
    )
  
    # 计算余弦相似度 [n_candidate, n_corpus]
    sim = encoder.similarity(candidate_feature, corpus_feature)
  
    # 加权求和得到最终得分
    scores = (sim * time_decay_weight).sum(axis=1) * 10  # [n_candidate]
  
    # 赋值并排序
    for s, c in zip(scores, candidate):
        c.score = s.item()
  
    candidate = sorted(candidate, key=lambda x: x.score, reverse=True)
    return candidate
