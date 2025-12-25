"""
Paper recommendation module using Qwen3 embeddings and tag-based scoring.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

from paper import ArxivPaper
from tag_scorer import extract_user_tags, calculate_tag_score


def rerank_paper(
    candidate: list[ArxivPaper],
    corpus: list[dict],
    model: str = 'Qwen/Qwen3-Embedding-0.6B',
    tag_weight: float = 0.3
) -> list[ArxivPaper]:
    """
    使用 Qwen3 嵌入相似度 + 标签匹配的混合推荐算法。
  
    算法流程：
    1. 提取用户 Zotero 库中的所有标签作为兴趣信号
    2. 使用 Qwen3 计算候选论文与用户收藏论文的语义相似度
    3. 计算候选论文与用户高频标签的匹配分数
    4. 加权融合两个分数，得到最终排序
  
    Args:
        candidate: arXiv 候选论文列表（待排序）
        corpus: Zotero 用户收藏的论文列表（用户兴趣）
        model: Qwen3 嵌入模型名称
        tag_weight: 标签分数权重 (0-1)，默认 0.3
  
    Returns:
        按综合得分降序排列的候选论文列表
    """
    # ===== 初始化模型 =====
    # Qwen3-Embedding 需要 transformers>=4.51.0, sentence-transformers>=2.7.0
    encoder = SentenceTransformer(model, trust_remote_code=True)
  
    # ===== Step 1: 提取用户标签 =====
    user_tags = extract_user_tags(corpus)
    print(f"[Tags] Extracted {len(user_tags)} unique tags from user library")
    if user_tags:
        print(f"[Tags] Top 10: {user_tags.most_common(10)}")
  
    # ===== Step 2: 准备 Corpus 数据 =====
    # 按添加日期排序，最新的在前
    corpus = sorted(
        corpus,
        key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
        reverse=True
    )
  
    # 时间衰减权重：越新的论文权重越高
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
  
    # ===== Step 3: Qwen3 嵌入计算（官方最佳实践）=====
    # Corpus（用户收藏）= Document，不使用 prompt
    corpus_abstracts = [paper['data']['abstractNote'] for paper in corpus]
    corpus_feature = encoder.encode(
        corpus_abstracts,
        batch_size=8,
        show_progress_bar=True
    )
  
    # Candidate（新论文）= Query，使用 prompt_name="query"
    candidate_summaries = [paper.summary for paper in candidate]
    candidate_feature = encoder.encode(
        candidate_summaries,
        prompt_name="query",  # 关键：Qwen3 官方推荐 query 使用 prompt
        batch_size=8,
        show_progress_bar=True
    )
  
    # ===== Step 4: 计算嵌入相似度分数 =====
    sim = encoder.similarity(candidate_feature, corpus_feature)  # [n_candidate, n_corpus]
    embedding_scores = (sim * time_decay_weight).sum(axis=1)
  
    # 转为 numpy 数组
    if hasattr(embedding_scores, 'numpy'):
        embedding_scores = embedding_scores.numpy()
    else:
        embedding_scores = np.array(embedding_scores)
  
    # ===== Step 5: 计算标签匹配分数 =====
    tag_scores = np.array([
        calculate_tag_score(f"{paper.title} {paper.summary}", user_tags)
        for paper in candidate
    ])
  
    # ===== Step 6: 归一化并融合分数 =====
    # 归一化嵌入分数到 [0, 1]
    emb_max = embedding_scores.max()
    if emb_max > 0:
        embedding_scores_norm = embedding_scores / emb_max
    else:
        # Edge case: if all embedding scores are 0, use zeros
        embedding_scores_norm = np.zeros_like(embedding_scores)
  
    # tag_scores 已经在 [0, 1] 范围内
  
    # 加权融合
    final_scores = (1 - tag_weight) * embedding_scores_norm + tag_weight * tag_scores
    final_scores = final_scores * 10  # 放大到合理显示范围
  
    # ===== Step 7: 赋值并排序 =====
    for score, paper in zip(final_scores, candidate):
        paper.score = float(score)
  
    candidate = sorted(candidate, key=lambda x: x.score, reverse=True)
  
    # 输出调试信息
    print(f"\n[Rerank] Completed. Top 5 recommendations:")
    for i, paper in enumerate(candidate[:5]):
        print(f"  {i+1}. [score={paper.score:.3f}] {paper.title[:70]}...")
  
    return candidate
