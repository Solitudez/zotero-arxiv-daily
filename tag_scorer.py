"""
Tag-based scoring module for paper recommendation.
Extracts user interest signals from Zotero tags.
"""
from collections import Counter


def extract_user_tags(corpus: list[dict]) -> Counter:
    """
    从用户 Zotero 库提取所有标签及其频率。
  
    Zotero API 返回的 tags 结构：
    item['data']['tags'] = [{'tag': 'NLP'}, {'tag': 'Transformer'}, ...]
  
    Args:
        corpus: Zotero API 返回的论文列表
  
    Returns:
        Counter: 标签(小写) -> 出现次数
    """
    tag_counter = Counter()
    for paper in corpus:
        tags = paper.get('data', {}).get('tags', [])
        for tag_obj in tags:
            tag = tag_obj.get('tag', '').lower().strip()
            if tag and len(tag) >= 2:  # 忽略太短的标签
                tag_counter[tag] += 1
    return tag_counter


def calculate_tag_score(paper_text: str, tag_counter: Counter, top_k: int = 50) -> float:
    """
    计算候选论文与用户高频标签的匹配分数。
  
    Args:
        paper_text: 候选论文的 title + summary
        tag_counter: 用户标签频率统计
        top_k: 只考虑前 k 个高频标签
  
    Returns:
        float: 匹配分数 (0.0 - 1.0)
    """
    if not tag_counter:
        return 0.0
  
    paper_text_lower = paper_text.lower()
    top_tags = tag_counter.most_common(top_k)
    total_weight = sum(count for _, count in top_tags)
  
    if total_weight == 0:
        return 0.0
  
    score = 0.0
    for tag, count in top_tags:
        # Use word boundary matching to avoid false positives
        # E.g., 'ml' won't match 'html', but will match 'ml' or 'ML'
        import re
        # Create pattern with word boundaries
        pattern = r'\b' + re.escape(tag) + r'\b'
        if re.search(pattern, paper_text_lower):
            score += count
  
    return score / total_weight
