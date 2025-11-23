from typing import List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEReranker:
    """
    基于 Cross-Encoder 的重排序器。
    不同于 Bi-Encoder (Retrieval)，Cross-Encoder 同时输入 (Query, Doc) 对，
    可以计算更精准的相关性分数，但计算开销大，适合对 Top-N 结果进行重排。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        """
        初始化 Reranker 模型。
        建议模型: 'BAAI/bge-reranker-v2-m3' 或 'BAAI/bge-reranker-large'
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading Reranker model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def compute_score(self, query: str, doc_text: str) -> float:
        """计算单对 (Query, Document) 的相关性分数"""
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs=[[query, doc_text]],
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)

            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            # BGE Reranker 输出的是 Logits，直接作为分数即可 (越大越好)
            return float(scores[0])

    def rerank(self, query: str, docs: List[str], doc_ids: List[str]) -> List[Tuple[str, float, int]]:
        """
        批量重排序。

        Args:
            query: 查询文本
            docs: 文档内容列表
            doc_ids: 文档ID列表 (用于对应返回结果)

        Returns:
            List[Tuple[doc_id, score, original_index]]: 按分数降序排列的结果
        """
        if not docs:
            return []

        # 构造 Query-Doc 对
        pairs = [[query, doc] for doc in docs]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)

            # 计算分数
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = scores.cpu().numpy()

        # 组合结果: (doc_id, score, original_index)
        results = []
        for i, score in enumerate(scores):
            results.append((doc_ids[i], float(score), i))

        # 按分数降序排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results
