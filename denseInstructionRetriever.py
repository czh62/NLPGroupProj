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


class Qwen3Retriever:
    """
    基于 Qwen3-Embedding (Large) 的稠密检索器。
    特点：
    1. 高维向量 (4096维)。
    2. 需要区分 Query 和 Document 的 Prompt 指令。
    """

    def __init__(self):
        self.doc_ids: List[str] = []
        self.documents: List[str] = []
        self.index: faiss.IndexFlatIP = None
        self.is_fitted = False

    def _get_embedding(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        核心工具方法：调用 Ollama 接口获取单个文本的 L2 归一化向量。
        关键点：根据 is_query 添加不同的 Instruction 前缀。
        """
        # 根据 Qwen 文档，Query 和 Doc 通常需要不同的指令前缀
        instruction = config.QWEN_QUERY_INSTRUCTION if is_query else config.QWEN_DOC_INSTRUCTION

        full_prompt = instruction + text
        payload = {
            "model": config.QWEN_MODEL_NAME,
            "prompt": full_prompt
        }
        try:
            resp = requests.post(config.OLLAMA_API_URL, json=payload, timeout=120)
            resp.raise_for_status()

            emb = resp.json()["embedding"]

            # 维度检查
            if len(emb) != config.QWEN_VECTOR_DIM:
                # 仅警告，有些量化版本可能会改变维度，但通常不会
                pass

            # 转换为 numpy 数组 (dtype=np.float32)，注意外部列表是必须的，以保持 (1 x Dim) 形状
            embedding_np = np.array([emb], dtype=np.float32)

            # 归一化以适配余弦相似度检索
            faiss.normalize_L2(embedding_np)

            # 返回形状为 (1 x Dim) 的数组
            return embedding_np

        except Exception as e:
            print(f"Ollama embedding error for text segment: {text[:30]}...")
            print(f"Error: {e}")
            raise

    def fit(self, documents: List[str], doc_ids: List[str] = None):
        """
        构建索引：对文档库进行 Embedding 并存入 FAISS。
        """
        self.documents = documents
        self.doc_ids = doc_ids if doc_ids else [str(i) for i in range(len(documents))]

        print(f"Total documents: {len(documents)}. Generating Qwen3 embeddings...")

        doc_vectors = []
        # 注意：Document 侧通常不需要特殊指令，或使用空指令 (is_query=False)
        for doc in tqdm(documents, desc="Vectorizing Documents"):
            try:
                # 获取单个文档向量 (shape: 1 x Dim) 并追加到列表中
                vector = self._get_embedding(doc, is_query=False)
                doc_vectors.append(vector)
            except Exception:
                # 遇到错误时停止处理
                raise

        # 将所有文档向量堆叠成一个 NumPy 数组 (shape: N x Dim)
        doc_vectors_np = np.vstack(doc_vectors)

        print("Building FAISS IndexFlatIP...")
        self.index = faiss.IndexFlatIP(config.QWEN_VECTOR_DIM)
        self.index.add(doc_vectors_np)

        self.is_fitted = True
        print("Qwen3 Retriever fitted successfully!")

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        检索方法：对查询进行 Embedding (带指令) 并搜索。
        """
        if not self.is_fitted:
            raise ValueError("Qwen3Retriever not fitted yet.")

        query_vec = self._get_embedding(query_text, is_query=True)

        # 搜索 Top-K
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0] * 100):
            if idx == -1:
                continue
            results.append((self.doc_ids[idx], float(score)))
        return results

    def save_index(self, dirpath: str):
        """保存模型和元数据"""
        os.makedirs(dirpath, exist_ok=True)

        print("Saving meta data...")
        with open(os.path.join(dirpath, "doc_ids.pkl"), "wb") as f:
            pickle.dump(self.doc_ids, f)

        print("Saving FAISS index...")
        faiss.write_index(self.index, os.path.join(dirpath, "faiss.index"))

        # 可选：保存原文以便调试
        with open(os.path.join(dirpath, "documents.jsonl"), "w", encoding="utf-8") as f:
            for doc_id, text in zip(self.doc_ids, self.documents):
                f.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")

        print(f"Index saved to {dirpath}")

    def load_index(self, dirpath: str):
        """加载模型和元数据"""
        ids_path = os.path.join(dirpath, "doc_ids.pkl")
        idx_path = os.path.join(dirpath, "faiss.index")

        if not (os.path.exists(ids_path) and os.path.exists(idx_path)):
            raise FileNotFoundError(f"Index files not found in {dirpath}")

        with open(ids_path, "rb") as f:
            self.doc_ids = pickle.load(f)

        self.index = faiss.read_index(idx_path)
        self.is_fitted = True
        print(f"Index loaded from {dirpath}, documents count: {len(self.doc_ids)}")


def main():
    """
    主程序：Qwen3 检索器的构建、加载与批量测试。
    """
    data_loader = HQSmallDataLoader(config.BASE_DATA_DIR)
    retriever = Qwen3Retriever()

    # 1. 索引管理
    index_file_check = os.path.join(config.QWEN_INDEX_DIR, "faiss.index")
    if os.path.exists(index_file_check):
        print(">>> Found existing Qwen3 index, loading...")
        retriever.load_index(config.QWEN_INDEX_DIR)
    else:
        print(">>> No index found, building from scratch...")
        doc_ids, documents = data_loader.load_collection(config.COLLECTION_PATH)
        retriever.fit(documents, doc_ids)
        retriever.save_index(config.QWEN_INDEX_DIR)

    # 2. 单条测试
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
    print(f"Saving predictions to {config.QWEN_OUTPUT_PATH}...")
    with open(config.QWEN_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for result in tqdm(batch_results, desc="Writing file"):
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(">>> Done.")


if __name__ == "__main__":
    main()