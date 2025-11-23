from typing import List, Tuple
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import config


class BGEReranker:
    """
    基于 Cross-Encoder 的重排序器 (分数统一归一化到 0-100)。
    """

    def __init__(self,
                 model_name: str = config.BGE_RERANKER_MODEL_NAME,
                 device: str = None,
                 api_key: str = None,
                 api_url: str = config.SF_API_URL):
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url

        if self.api_key:
            self.use_api = True
            print(f"Using Reranker API Mode: {self.model_name}")
            self.tokenizer = None
            self.model = None
        else:
            self.use_api = False
            self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading Local Reranker model: {self.model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def rerank(self, query: str, docs: List[str], doc_ids: List[str]) -> List[Tuple[str, float, int]]:
        if not docs:
            return []
        if len(docs) != len(doc_ids):
            raise ValueError("docs 和 doc_ids 长度必须一致")

        if self.use_api:
            return self._rerank_via_api(query, docs, doc_ids)
        else:
            return self._rerank_via_local(query, docs, doc_ids)

    def _rerank_via_local(self, query: str, docs: List[str], doc_ids: List[str]) -> List[Tuple[str, float, int]]:
        """本地模型推理逻辑"""
        pairs = [[query, doc] for doc in docs]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)

            logits = self.model(**inputs, return_dict=True).logits.view(-1, ).float()

            # 本地 Sigmoid (0-1) -> 映射到 0-100
            scores = torch.sigmoid(logits) * 100

            scores = scores.cpu().numpy()

        results = []
        for i, score in enumerate(scores):
            results.append((doc_ids[i], float(score), i))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _rerank_via_api(self, query: str, docs: List[str], doc_ids: List[str]) -> List[Tuple[str, float, int]]:
        """API 调用逻辑"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": docs,
            "return_documents": False
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            resp_data = response.json()

            if "results" not in resp_data:
                print(f"API Error: Unexpected response format: {resp_data}")
                return []

            results = []
            for item in resp_data["results"]:
                idx = item["index"]
                raw_score = item["relevance_score"]

                final_score = raw_score * 100

                results.append((doc_ids[idx], float(final_score), idx))

            results.sort(key=lambda x: x[1], reverse=True)
            return results

        except requests.exceptions.RequestException as e:
            print(f"API Request Failed: {e}")
            return []