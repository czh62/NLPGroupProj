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
    支持本地 Ollama 推理或 SiliconFlow API 远程批量推理。
    """

    def __init__(self, api_key: str = None):
        self.doc_ids: List[str] = []
        self.documents: List[str] = []
        self.index: faiss.IndexFlatIP = None
        self.is_fitted = False
        self.api_key = api_key

    def _get_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        核心工具方法：获取文本列表的 L2 归一化向量。
        逻辑：
        1. 如果 config 中有 API KEY，则使用 SiliconFlow API 进行 Batch 推理。
        2. 否则使用 Ollama 进行逐个推理并拼接。
        """
        # 1. 预处理：根据 Qwen 文档添加指令前缀
        instruction = config.QWEN_QUERY_INSTRUCTION if is_query else config.QWEN_DOC_INSTRUCTION
        processed_texts = [instruction + text for text in texts]


        # 容器用于存放原始向量列表
        raw_embeddings = []

        # ==================== 分支 A: 使用 SiliconFlow API (Batch) ====================
        if self.api_key:
            url = config.SF_API_EMBEDDING_URL
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": config.SF_QWEN_MODEL_NAME,  # 确保 config 中的模型名也是 SiliconFlow 支持的 ID
                "input": processed_texts,
                "encoding_format": "float"
            }

            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=120)
                resp.raise_for_status()
                resp_json = resp.json()

                # 假设 OpenAI 格式: {"data": [{"embedding": [...]}, ...]}
                # 按照 index 排序确保顺序一致
                data_items = sorted(resp_json["data"], key=lambda x: x["index"])
                raw_embeddings = [item["embedding"] for item in data_items]

            except Exception as e:
                print(f"SiliconFlow API error: {e}")
                # 这里可以选择抛出异常或者降级，这里选择抛出
                raise

        # ==================== 分支 B: 使用 Ollama 本地 (Iterative) ====================
        else:
            # 逐个调用 Ollama
            for text in processed_texts:
                payload = {
                    "model": config.QWEN_MODEL_NAME,
                    "prompt": text
                }
                try:
                    resp = requests.post(config.OLLAMA_API_URL, json=payload, timeout=120)
                    resp.raise_for_status()
                    emb = resp.json()["embedding"]
                    raw_embeddings.append(emb)
                except Exception as e:
                    print(f"Ollama embedding error for text segment: {text[:30]}...")
                    raise e

        # ==================== 后处理：转 Numpy + 归一化 ====================

        if not raw_embeddings:
            raise ValueError("No embeddings were generated.")

        # 转换为 (Batch_Size, Dim)
        embeddings_np = np.array(raw_embeddings, dtype=np.float32)

        # 维度检查 (取第一行检查)
        if embeddings_np.shape[1] != config.QWEN_VECTOR_DIM:
            # 仅打印警告，防止不同量化版本维度差异
            pass

        # 归一化 (In-place L2 normalization for Cosine Similarity)
        faiss.normalize_L2(embeddings_np)

        return embeddings_np

    def fit(self, documents: List[str], doc_ids: List[str] = None):
        """
        构建索引：对文档库进行 Embedding 并存入 FAISS。
        使用 Batch 处理以适配 _get_embeddings 的 API 优势。
        """
        self.documents = documents
        self.doc_ids = doc_ids if doc_ids else [str(i) for i in range(len(documents))]

        print(f"Total documents: {len(documents)}. Generating embeddings...")

        doc_vectors_list = []

        # 定义 Batch Size (API 模式建议大一点，比如 32-64；Ollama 模式无所谓，内部是串行的)
        # 如果使用 SiliconFlow，建议设为 16 ~ 64，取决于文本长度限制

        # 使用 tqdm 显示进度，步长为 batch_size
        for i in tqdm(range(0, len(documents), config.BATCH_SIZE), desc="Vectorizing Batches"):
            batch_docs = documents[i: i + config.BATCH_SIZE]
            try:
                # 调用新方法，批量获取向量 (N x Dim)
                batch_vectors = self._get_embeddings(batch_docs, is_query=False)
                doc_vectors_list.append(batch_vectors)
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                raise

        # 将所有 Batch 的结果堆叠成一个大数组 (Total_N x Dim)
        if doc_vectors_list:
            all_vectors_np = np.vstack(doc_vectors_list)
        else:
            raise ValueError("No documents were vectorized.")

        print(f"Building FAISS IndexFlatIP with shape {all_vectors_np.shape}...")
        self.index = faiss.IndexFlatIP(config.QWEN_VECTOR_DIM)
        self.index.add(all_vectors_np)

        self.is_fitted = True
        print("Qwen3 Retriever fitted successfully!")

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        检索方法：对查询进行 Embedding (带指令) 并搜索。
        """
        if not self.is_fitted:
            raise ValueError("Qwen3Retriever not fitted yet.")

        # 将单个 Query 放入列表调用 _get_embeddings
        # 返回 shape (1, Dim)
        query_vec = self._get_embeddings([query_text], is_query=True)

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

        # 保存原文
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


    # 检查 API Key 状态
    if getattr(config, "SF_API_KEY", None):
        print(f">>> Using SiliconFlow API for embeddings (Model: {config.SF_QWEN_MODEL_NAME})")
        retriever = Qwen3Retriever(api_key = config.SF_API_KEY)
    else:
        print(f">>> Using Local Ollama for embeddings (Model: {config.QWEN_MODEL_NAME})")
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