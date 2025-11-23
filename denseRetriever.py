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
    支持本地 Ollama 逐个推理或 SiliconFlow API 远程批量推理。
    """

    def __init__(self,  api_key: str = None):
        self.doc_ids: List[str] = []
        self.documents: List[str] = []
        self.index: faiss.IndexFlatIP = None
        self.is_fitted = False
        self.api_key = api_key

    def _get_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        核心工具方法：获取文本列表的 L2 归一化向量。
        """
        # BGE 通常不需要指令，但如果需要，可以从 config 中读取
        # BGE 文档通常推荐在查询前添加 '为这个句子生成表示用于检索相关文档:'
        if is_query and getattr(config, "BGE_QUERY_INSTRUCTION", None):
            instruction = config.BGE_QUERY_INSTRUCTION
            processed_texts = [instruction + text for text in texts]
        else:
            processed_texts = texts

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
                "model": config.SF_BGE_MODEL_NAME,  # 假设 config 中有对应的 BGE 模型名
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
                # 抛出异常
                raise

        # ==================== 分支 B: 使用 Ollama 本地 (Iterative) ====================
        else:
            # 逐个调用 Ollama
            for text in processed_texts:
                payload = {
                    "model": config.BGE_MODEL_NAME,
                    "prompt": text
                }
                try:
                    # 使用较短的超时时间，因为是单次请求
                    resp = requests.post(config.OLLAMA_API_URL, json=payload, timeout=60)
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
        if embeddings_np.shape[1] != config.BGE_VECTOR_DIM:
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

        # 确定使用的模型名称以便打印
        model_name = config.SF_BGE_MODEL_NAME if self.api_key else config.BGE_MODEL_NAME
        print(f"Total documents: {len(documents)}. Generating embeddings using {model_name}...")

        doc_vectors_list = []

        # 使用 tqdm 显示进度，步长为 config.BATCH_SIZE
        for i in tqdm(range(0, len(documents), config.BATCH_SIZE), desc="Vectorizing Batches"):
            batch_docs = documents[i: i + config.BATCH_SIZE]
            try:
                # 调用批量方法，批量获取向量 (N x Dim)
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
        self.index = faiss.IndexFlatIP(config.BGE_VECTOR_DIM)
        self.index.add(all_vectors_np)

        self.is_fitted = True
        print("BGE Retriever fitted successfully!")

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        检索：将 Query 向量化后在索引中查找最近邻。
        """
        if not self.is_fitted:
            raise ValueError("BGERetriever not fitted yet.")

        # 将单个 Query 放入列表调用 _get_embeddings
        # 返回 shape (1, Dim)
        query_vec = self._get_embeddings([query_text], is_query=True)

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


    # 检查 API Key 状态并打印当前模式
    if getattr(config, "SF_API_KEY", None):
        print(f">>> Using SiliconFlow API for embeddings (Model: {config.SF_BGE_MODEL_NAME})")
        retriever = BGERetriever(api_key = config.SF_API_KEY)
    else:
        print(f">>> Using Local Ollama for embeddings (Model: {config.BGE_MODEL_NAME})")
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