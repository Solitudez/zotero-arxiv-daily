"""
Paper recommendation module using GGUF Qwen3-Embedding-4B.

这是 recommender.py 的 GGUF 版本，使用 llama-cpp-python 加载
Qwen3-Embedding-4B 量化模型进行嵌入计算。

与原版的主要区别：
1. 使用 GGUFEmbedder 替代 SentenceTransformer
2. 使用 is_query=True 替代 prompt_name="query"
3. 使用自定义 similarity() 替代 encoder.similarity()
"""

import numpy as np
from datetime import datetime
from paper import ArxivPaper
from tag_scorer import extract_user_tags, calculate_tag_score
from gguf_embedder import GGUFEmbedder
from loguru import logger


def rerank_paper_gguf(
    candidate: list[ArxivPaper],
    corpus: list[dict],
    tag_weight: float = 0.3
) -> list[ArxivPaper]:
    """
    使用 GGUF Qwen3-Embedding-4B 嵌入相似度 + 标签匹配的混合推荐算法。
    
    算法流程：
    1. 提取用户 Zotero 库中的所有标签作为兴趣信号
    2. 使用 Qwen3-Embedding-4B (GGUF) 计算候选论文与用户收藏论文的语义相似度
    3. 计算候选论文与用户高频标签的匹配分数
    4. 加权融合两个分数，得到最终排序
    
    Args:
        candidate: arXiv 候选论文列表（待排序）
        corpus: Zotero 用户收藏的论文列表（用户兴趣）
        tag_weight: 标签分数权重 (0-1)，默认 0.3
    
    Returns:
        按综合得分降序排列的候选论文列表
    """
    
    # 添加空 corpus 检查
    if not corpus:
        logger.warning("Empty corpus, returning original candidate order")
        return candidate
    
    # ===== 初始化 GGUF 嵌入模型 =====
    encoder = GGUFEmbedder()
    
    # ===== Step 1: 提取用户标签 =====
    user_tags = extract_user_tags(corpus)
    logger.info(f"[Tags] Extracted {len(user_tags)} unique tags from user library")
    if user_tags:
        logger.info(f"[Tags] Top 10: {user_tags.most_common(10)}")
    
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
    
    # ===== Step 3: GGUF 嵌入计算 =====
    # Corpus（用户收藏）= Document，不使用 query 前缀
    corpus_abstracts = [paper['data']['abstractNote'] for paper in corpus]
    corpus_feature = encoder.encode(
        corpus_abstracts,
        is_query=False,  # Document 端
        show_progress_bar=True
    )
    
    # Candidate（新论文）= Query，使用 query 前缀
    candidate_summaries = [paper.summary for paper in candidate]
    candidate_feature = encoder.encode(
        candidate_summaries,
        is_query=True,  # Query 端，会自动添加指令前缀
        show_progress_bar=True
    )
    
    # ===== Step 4: 计算嵌入相似度分数 =====
    sim = encoder.similarity(candidate_feature, corpus_feature)  # [n_candidate, n_corpus]
    embedding_scores = (sim * time_decay_weight).sum(axis=1)
    
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
        embedding_scores_norm = np.zeros_like(embedding_scores)
    
    # 加权融合
    final_scores = (1 - tag_weight) * embedding_scores_norm + tag_weight * tag_scores
    final_scores = final_scores * 10  # 放大到合理显示范围
    
    # ===== Step 7: 赋值并排序 =====
    for score, paper in zip(final_scores, candidate):
        paper.score = float(score)
    
    candidate = sorted(candidate, key=lambda x: x.score, reverse=True)
    
    # 输出调试信息
    logger.info(f"\n[Rerank-GGUF] Completed. Top 5 recommendations:")
    for i, paper in enumerate(candidate[:5]):
        logger.info(f"  {i+1}. [score={paper.score:.3f}] {paper.title[:70]}...")
    
    return candidate
