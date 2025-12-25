import numpy as np
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime

def rerank_paper(candidate:list[ArxivPaper],corpus:list[dict],model:str='Qwen/Qwen3-Embedding-0.6B') -> list[ArxivPaper]:
    # Qwen3-Embedding-0.6B 需要 transformers>=4.51.0 和 sentence-transformers>=2.7.0
    encoder = SentenceTransformer(
        model,
        trust_remote_code=True,
        # 以下参数可选，用于在有 GPU 和 flash_attention 时加速
        # model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
        # tokenizer_kwargs={"padding_side": "left"},
    )
    #sort corpus by date, from newest to oldest
    corpus = sorted(corpus,key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in corpus])
    candidate_feature = encoder.encode([paper.summary for paper in candidate])
    sim = encoder.similarity(candidate_feature,corpus_feature) # [n_candidate, n_corpus]
    scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
    for s,c in zip(scores,candidate):
        c.score = s.item()
    candidate = sorted(candidate,key=lambda x: x.score,reverse=True)
    return candidate
