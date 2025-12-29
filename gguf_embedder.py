"""
GGUF 格式嵌入模型封装模块

使用 llama-cpp-python 加载 Qwen3-Embedding-4B GGUF 模型，
提供与 sentence-transformers 类似的 API 接口。

Model: Casual-Autopsy/Qwen3-Embedding-4B-GGUFs
File: Qwen3-Embedding-4B-q4_k_m.gguf
"""

from llama_cpp import Llama
import numpy as np
from numpy.linalg import norm
from typing import Optional
from loguru import logger


class GGUFEmbedder:
    """
    GGUF 格式 Qwen3-Embedding 模型封装类
    
    特点：
    - 使用 llama-cpp-python 加载 GGUF 模型
    - 支持 Qwen3-Embedding 的指令格式（query 需要添加前缀）
    - 提供 encode() 和 similarity() 方法，API 风格类似 sentence-transformers
    """
    
    # 默认模型配置
    DEFAULT_REPO = "Casual-Autopsy/Qwen3-Embedding-4B-GGUFs"
    DEFAULT_FILE = "Qwen3-Embedding-4B-q4_k_m.gguf"
    
    # Qwen3-Embedding 官方推荐的检索任务指令
    DEFAULT_TASK = "Given a web search query, retrieve relevant passages that answer the query"
    
    def __init__(
        self,
        repo_id: str = DEFAULT_REPO,
        filename: str = DEFAULT_FILE,
        n_ctx: int = 8192,
        n_threads: int = 4,
        verbose: bool = False
    ):
        """
        初始化 GGUF 嵌入模型
        
        Args:
            repo_id: HuggingFace 模型仓库 ID
            filename: GGUF 模型文件名
            n_ctx: 上下文窗口大小
            n_threads: CPU 线程数（GitHub Actions 建议设为 4）
            verbose: 是否显示详细日志
        """
        logger.info(f"Loading GGUF embedding model: {repo_id}/{filename}")
        
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            embedding=True,  # 关键：启用 embedding 模式
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose
        )
        self.task = self.DEFAULT_TASK
        
        logger.info("GGUF embedding model loaded successfully")
    
    def _format_query(self, text: str) -> str:
        """
        格式化查询文本，添加 Qwen3-Embedding 所需的指令前缀
        
        Qwen3-Embedding 使用非对称检索：
        - Query 端需要添加 "Instruct: {task}\nQuery:{text}" 前缀
        - Document 端不需要添加前缀
        """
        return f'Instruct: {self.task}\nQuery:{text}'
    
    def encode(
        self,
        texts: list[str],
        is_query: bool = False,
        show_progress_bar: bool = True,
        batch_size: int = 1,  # GGUF 不支持真正的 batch，保留参数兼容
        prompt_name: Optional[str] = None  # 兼容 sentence-transformers API
    ) -> np.ndarray:
        """
        生成文本的嵌入向量
        
        Args:
            texts: 待编码的文本列表
            is_query: 是否为查询文本（需要添加指令前缀）
            show_progress_bar: 是否显示进度条
            batch_size: 批处理大小（GGUF 实际逐个处理）
            prompt_name: 兼容参数，如果为 "query" 则等同于 is_query=True
            
        Returns:
            numpy 数组，shape: (len(texts), embedding_dim)
        """
        # 兼容 sentence-transformers 的 prompt_name 参数
        if prompt_name == "query":
            is_query = True
        
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            # Query 需要添加指令前缀
            if is_query:
                text = self._format_query(text)
            
            # 截断过长文本（避免超出上下文限制）
            if len(text) > 8000:
                text = text[:8000]
            
            # 生成 embedding
            emb = self.llm.embed(text)
            embeddings.append(emb)
            
            # 显示进度
            if show_progress_bar and (i + 1) % 10 == 0:
                logger.info(f"Embedding progress: {i + 1}/{total}")
        
        if show_progress_bar:
            logger.info(f"Embedding completed: {total}/{total}")
        
        return np.array(embeddings)
    
    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        计算两组向量之间的余弦相似度矩阵
        
        Args:
            a: 第一组向量，shape: (n, dim)
            b: 第二组向量，shape: (m, dim)
            
        Returns:
            相似度矩阵，shape: (n, m)
        """
        # L2 归一化
        a_norm = a / (norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (norm(b, axis=1, keepdims=True) + 1e-8)
        
        # 计算余弦相似度
        return np.dot(a_norm, b_norm.T)
