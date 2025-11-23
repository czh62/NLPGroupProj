import json
import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
import requests
from tqdm import tqdm

import config
from HQSmallDataLoader import HQSmallDataLoader


class BGERetriever:
    """
    基于 BGE-M3 (BAAI General Embedding) 的稠密检索器。
    原理：
    1. 调用本地 Ollama API 获取文本的 Embedding 向量。
    2. 使用 FAISS 进行向量索引和检索。
    """

    def __init__(self):
        self.doc_ids: List[str] = []
        self.documents: List[str] = []
        self.index: faiss.IndexFlatIP = None  # BGE 推荐使用内积（归一化后等同于余弦相似度）
        self.is_fitted = False

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        核心工具方法：调用 Ollama 接口获取单个文本的向量。
        """
        # 构造请求体
        payload = {
            "model": config.BGE_MODEL_NAME,
            "prompt": text
        }
        try:
            resp = requests.post(config.OLLAMA_API_URL, json=payload, timeout=60)
            resp.raise_for_status()

            # 解析响应
            vec = resp.json()["embedding"]

            # 维度校验
            if len(vec) != config.BGE_VECTOR_DIM:
                print(f"Warning: Expected dim {config.BGE_VECTOR_DIM}, got {len(vec)}")

            # 转换为 numpy 数组 (dtype=np.float32)
            embedding_np = np.array([vec], dtype=np.float32)

            # 进行 L2 归一化 (BGE 模型的标准操作)
            faiss.normalize_L2(embedding_np)

            # 返回形状为 (1 x Dim) 的数组
            return embedding_np

        except Exception as e:
            print(f"Embedding request failed for text segment: {text[:30]}...")
            print(f"Error: {e}")
            raise

    def fit(self, documents: List[str], doc_ids: List[str] = None):
        """
        构建索引：遍历所有文档生成向量，并插入 FAISS。
        """
        self.documents = documents
        self.doc_ids = doc_ids if doc_ids else [str(i) for i in range(len(documents))]

        print(f"Generating embeddings for {len(documents)} documents using {config.BGE_MODEL_NAME}...")

        doc_vectors = []
        for doc in tqdm(documents, desc="Vectorizing Documents"):
            try:
                # 获取单个文档向量 (shape: 1 x Dim) 并追加到列表中
                vector = self._get_embedding(doc)
                doc_vectors.append(vector)
            except Exception:
                # 遇到错误时停止处理
                raise

        # 将所有文档向量堆叠成一个 NumPy 数组 (shape: N x Dim)
        doc_vectors_np = np.vstack(doc_vectors)

        print("Building FAISS IndexFlatIP...")
        self.index = faiss.IndexFlatIP(config.BGE_VECTOR_DIM)
        self.index.add(doc_vectors_np)

        self.is_fitted = True
        print("BGE Retriever fitted successfully!")

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        检索：将 Query 向量化后在索引中查找最近邻。
        """
        if not self.is_fitted:
            raise ValueError("BGERetriever not fitted yet.")

        # 获取查询向量 (shape: 1 x Dim)
        query_vec = self._get_embedding(query_text)

        # FAISS 搜索
        # query_vec 已经是 (1 x Dim) 的 NumPy 数组，可以直接用于搜索
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0] * 100):
            if idx == -1:
                continue
            results.append((self.doc_ids[idx], float(score)))
        return results

    def save_index(self, dirpath: str):
        """保存索引数据"""
        os.makedirs(dirpath, exist_ok=True)

        print("Saving doc_ids...")
        with open(os.path.join(dirpath, "doc_ids.pkl"), "wb") as f:
            pickle.dump(self.doc_ids, f)

        print("Saving FAISS index...")
        faiss.write_index(self.index, os.path.join(dirpath, "faiss.index"))

        print("Saving original documents (optional for inspection)...")
        with open(os.path.join(dirpath, "documents.jsonl"), "w", encoding="utf-8") as f:
            for doc_id, text in zip(self.doc_ids, self.documents):
                f.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")

        print(f"Index saved to {dirpath}")

    def load_index(self, dirpath: str):
        """加载索引数据"""
        ids_path = os.path.join(dirpath, "doc_ids.pkl")
        idx_path = os.path.join(dirpath, "faiss.index")

        if not (os.path.exists(ids_path) and os.path.exists(idx_path)):
            raise FileNotFoundError(f"Index files not found in {dirpath}")

        print("Loading doc_ids...")
        with open(ids_path, "rb") as f:
            self.doc_ids = pickle.load(f)

        print("Loading FAISS index...")
        self.index = faiss.read_index(idx_path)

        self.is_fitted = True
        print(f"Index loaded, containing {len(self.doc_ids)} documents.")


def main():
    """
    主程序：BGE 检索器的构建、加载与批量测试。
    """
    data_loader = HQSmallDataLoader(config.BASE_DATA_DIR)
    retriever = BGERetriever()

    # 1. 索引管理
    index_file_check = os.path.join(config.BGE_INDEX_DIR, "faiss.index")
    if os.path.exists(index_file_check):
        print(">>> Found existing BGE index, loading...")
        retriever.load_index(config.BGE_INDEX_DIR)
    else:
        print(">>> No index found, building from scratch...")
        # 注意：这可能需要较长时间，取决于文档数量和显卡性能
        print("Loading document collection...")
        doc_ids, documents = data_loader.load_collection(config.COLLECTION_PATH)
        retriever.fit(documents, doc_ids)
        retriever.save_index(config.BGE_INDEX_DIR)

    # 2. 单条查询测试 (Sanity Check)
    sample_query = "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?"
    print(f"\n>>> Sanity Check Query: {sample_query}")
    results = retriever.query(sample_query, top_k=10)
    print("Top 10 results:")
    for i, (doc_id, score) in enumerate(results, 1):
        print(f"  {i}. DocID: {doc_id}, Score: {score:.4f}")

    # 3. 批量处理测试集 (Test Set)
    print(f"\n>>> Batch Processing Test Set from {config.VALIDATION_SET_PATH}...")
    test_queries_data = data_loader.load_test_set(config.VALIDATION_SET_PATH)

    batch_results = []
    for query_item in tqdm(test_queries_data, desc="Retrieving"):
        query_id = query_item["id"]
        query_text = query_item["text"]

        results = retriever.query(query_text, top_k=10)

        batch_results.append({
            "id": query_id,
            "question": query_text,
            "retrieved_docs": [[doc_id, float(score)] for doc_id, score in results]
        })

    # 4. 保存
    print(f"Saving predictions to {config.BGE_OUTPUT_PATH}...")
    with open(config.BGE_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for result in tqdm(batch_results, desc="Writing file"):
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(">>> Done.")


if __name__ == "__main__":
    main()