import json
import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
import requests
from tqdm import tqdm

from HQSmallDataLoader import HQSmallDataLoader

# ==================== 配置区 ====================
API_URL = "https://api.siliconflow.cn/v1/embeddings"
API_KEY = os.environ.get('SILICONFLOW_API_KEY') # ←←← 请替换为你的真实 key
print(API_KEY)
MODEL_NAME = "BAAI/bge-m3"  # 支持中英文，质量极高
BATCH_SIZE = 32  # 根据 API QPS 调整，SiliconFlow
VECTOR_DIM = 1024  # 维度
INDEX_DIR = "./data/bge_index"
COLLECTION_PATH = "./COMP5423-25Fall-HQ-small/collection.jsonl"


# ================================================


class BGERetriever:
    """
    基于 BAAI/bge-large-zh-v1.5 的 Dense Retriever
    使用 SiliconFlow 在线 embedding 接口 + FAISS 内积检索
    """

    def __init__(self):
        self.doc_ids: List[str] = []
        self.documents: List[str] = []
        self.index: faiss.IndexFlatIP = None
        self.is_fitted = False

    # ----------- 调用 SiliconFlow Embedding API -----------
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batch"):
            batch = texts[i:i + BATCH_SIZE]
            payload = {
                "model": MODEL_NAME,
                "input": batch,
                "encoding_format": "float"
            }
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            try:
                resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(embeddings)
            except Exception as e:
                print(f"Embedding request failed: {e}")
                print("Response:", resp.text if 'resp' in locals() else "None")
                raise

        embeddings_np = np.array(all_embeddings, dtype=np.float32)

        # L2 归一化（BGE 官方推荐用于内积相似度）
        faiss.normalize_L2(embeddings_np)
        return embeddings_np

    # ----------- FIT：构建索引 -----------
    def fit(self, documents: List[str], doc_ids: List[str] = None):
        self.documents = documents
        self.doc_ids = doc_ids if doc_ids else [str(i) for i in range(len(documents))]

        print(f"Total documents: {len(documents)}，开始获取 BGE 向量...")
        doc_vectors = self._get_embeddings(documents)

        print("Building FAISS IndexFlatIP...")
        self.index = faiss.IndexFlatIP(VECTOR_DIM)
        self.index.add(doc_vectors)  # 已归一化，直接加进去

        self.is_fitted = True
        print("BGE Retriever fitted successfully！")

    # ----------- QUERY -----------
    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.is_fitted:
            raise ValueError("BGERetriever not fitted yet.")

        print(f"Query embedding: {query_text}")
        query_vec = self._get_embeddings([query_text])  # shape: (1, 1024)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.doc_ids[idx], float(score)))  # score ∈ [-1, 1]，已归一化后接近余弦相似度

        return results

    # ----------- 保存索引 -----------
    def save_index(self, dirpath: str = INDEX_DIR):
        os.makedirs(dirpath, exist_ok=True)

        print("Saving doc_ids...")
        with open(os.path.join(dirpath, "doc_ids.pkl"), "wb") as f:
            pickle.dump(self.doc_ids, f)

        print("Saving FAISS index...")
        faiss.write_index(self.index, os.path.join(dirpath, "faiss.index"))

        # 可选：保存原始文档便于后续查看
        with open(os.path.join(dirpath, "documents.jsonl"), "w", encoding="utf-8") as f:
            for doc_id, text in zip(self.doc_ids, self.documents):
                f.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")

        print(f"Index saved to {dirpath}")

    # ----------- 加载索引 -----------
    def load_index(self, dirpath: str = INDEX_DIR):
        print("Loading doc_ids...")
        with open(os.path.join(dirpath, "doc_ids.pkl"), "rb") as f:
            self.doc_ids = pickle.load(f)

        print("Loading FAISS index...")
        self.index = faiss.read_index(os.path.join(dirpath, "faiss.index"))

        self.is_fitted = True
        print(f"Index loaded from {dirpath}，documents count: {len(self.doc_ids)}")


def main():
    # 1. 加载数据
    data_loader = HQSmallDataLoader("./data")
    index_dir = INDEX_DIR

    retriever = BGERetriever()

    if os.path.exists(f"{index_dir}/faiss.index"):
        print("Found existing BGE index，loading...")
        retriever.load_index(index_dir)
    else:
        print("No index found，building new BGE index...")
        doc_ids, docs = data_loader.load_collection(COLLECTION_PATH)
        retriever.fit(docs, doc_ids)
        retriever.save_index(index_dir)

    # 2. 测试查询（支持中英文混合）
    test_queries = [
        "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?",
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"Query: {q}")
        results = retriever.query(q, top_k=5)
        print("Top-5 Results:")
        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"  {rank:>2}. [DocID: {doc_id}]  Score: {score:.4f}")


if __name__ == "__main__":
    main()
