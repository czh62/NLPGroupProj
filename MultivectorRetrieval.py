import json
import os
import pickle
from typing import List, Tuple, Dict, Any, Optional

import faiss
import numpy as np
import requests
from tqdm import tqdm

import config
from HQSmallDataLoader import HQSmallDataLoader


class MultiVectorRetriever:
    """
    基于自定义切分和 Ollama 向量的 Multi-Vector 检索器。
    它将大的“父文档”切分成小的“子文档”进行索引，
    但检索时返回完整的父文档内容。
    """

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        # 1. 核心存储
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        # 存储 子文档 ID -> 父文档 ID 的映射
        self.sub_doc_id_to_parent_id: List[str] = []
        # 存储 父文档 ID -> 父文档内容的映射 (Docstore 功能)
        self.parent_id_to_content: Dict[str, str] = {}

        # 2. 配置
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.is_fitted = False
        self.ID_KEY = "parent_doc_id"  # 唯一ID键

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        核心工具方法：调用 Ollama 接口获取单个文本的 L2 归一化向量。
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

            # 转换为 numpy 数组 (dtype=np.float32)，注意外部列表是必须的，以保持 (1 x Dim) 形状
            embedding_np = np.array([vec], dtype=np.float32)

            # 进行 L2 归一化 (BGE/稠密检索器的标准操作)
            faiss.normalize_L2(embedding_np)

            # 返回形状为 (1 x Dim) 的数组
            return embedding_np

        except Exception as e:
            # 打印前 30 个字符
            print(f"Embedding request failed for text segment: {text[:30]}...")
            print(f"Error: {e}")
            raise  # 中断处理

    def _simple_text_splitter(self, text: str) -> List[str]:
        """简单的文本切分模拟 (LangChain RecursiveCharacterTextSplitter 的功能)"""
        if not text:
            return []

        chunks = []
        # 简单的滑动窗口切分
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
            if i + self.chunk_size >= len(text):
                break
        return chunks

    def fit(self, documents: List[str], doc_ids: List[str]):
        """
        构建索引：
        1. 存储父文档内容。
        2. 对每个父文档进行切分，生成子文档。
        3. 索引子文档的向量到 FAISS。
        **修改点：通过循环调用 _get_embedding 处理批量子文档**
        """
        print(f"Building Multi-Vector Index for {len(documents)} documents...")

        sub_documents: List[str] = []

        # 存储父文档内容
        for parent_id, content in zip(doc_ids, documents):
            self.parent_id_to_content[parent_id] = content

        # 切分并收集子文档
        print("Splitting documents into chunks (Sub-Documents)...")
        for parent_id, content in tqdm(zip(doc_ids, documents), total=len(documents), desc="Splitting"):
            chunks = self._simple_text_splitter(content)

            for chunk in chunks:
                # 收集子文档内容
                sub_documents.append(chunk)
                # 记录子文档到父文档的映射
                self.sub_doc_id_to_parent_id.append(parent_id)

        print(f"Total sub-documents (chunks) created: {len(sub_documents)}")
        print(f"Generating embeddings for {len(sub_documents)} sub-documents...")

        # 循环调用 _get_embedding
        sub_doc_vectors = []
        for doc in tqdm(sub_documents, desc="Vectorizing Sub-Documents"):
            try:
                # _get_embedding 返回 (1 x Dim) 数组
                vector = self._get_embedding(doc)
                sub_doc_vectors.append(vector)
            except Exception:
                raise

        # 将所有子文档向量堆叠成一个 NumPy 数组 (shape: N x Dim)
        sub_doc_vectors_np = np.vstack(sub_doc_vectors)

        print("Building FAISS IndexFlatIP for Sub-Documents...")
        self.faiss_index = faiss.IndexFlatIP(config.BGE_VECTOR_DIM)
        self.faiss_index.add(sub_doc_vectors_np)

        self.is_fitted = True
        print("Multi-Vector Retriever fitted successfully!")

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        检索：
        1. Query 向量化。
        2. 在 FAISS 索引中查找 top_k 个最相似的 子文档 ID。
        3. 利用映射找到对应的 父文档 ID。
        4. 去重，并返回 父文档内容。
        **修改点：调用 _get_embedding 获取单个查询向量**
        """
        if not self.is_fitted:
            raise ValueError("MultiVectorRetriever not fitted yet.")

        # 获取查询向量 (shape: 1 x Dim)
        query_vec = self._get_embedding(query_text)

        # FAISS 搜索 (检索的是子文档的索引)
        scores, indices = self.faiss_index.search(query_vec, top_k * 5)  # 检索更多，以便去重后有足够的父文档

        retrieved_parent_ids = {}  # 用于去重和记录最高分数 {parent_id: (score, sub_doc_idx)}

        for idx, score in zip(indices[0], scores[0] * 100):
            if idx == -1 or idx >= len(self.sub_doc_id_to_parent_id):
                continue

            parent_id = self.sub_doc_id_to_parent_id[idx]

            # 如果父文档 ID 还没有被记录，或者当前分数更高，则记录
            if parent_id not in retrieved_parent_ids or score > retrieved_parent_ids[parent_id][0]:
                retrieved_parent_ids[parent_id] = (score, idx)

        # 转换为最终结果格式 (父文档 ID, 父文档内容, 最高分数)
        results = []
        # 按分数降序排列
        sorted_results = sorted(retrieved_parent_ids.items(), key=lambda item: item[1][0], reverse=True)

        for parent_id, (score, _) in sorted_results[:top_k]:
            parent_content = self.parent_id_to_content.get(parent_id, "CONTENT NOT FOUND")
            results.append((parent_id, parent_content, float(score)))

        return results

    def save_index(self, dirpath: str):
        """保存索引数据"""
        os.makedirs(dirpath, exist_ok=True)

        print("Saving parent_id_to_content...")
        with open(os.path.join(dirpath, "parent_content.pkl"), "wb") as f:
            pickle.dump(self.parent_id_to_content, f)

        print("Saving sub_doc_id_to_parent_id mapping...")
        with open(os.path.join(dirpath, "sub_to_parent_ids.pkl"), "wb") as f:
            pickle.dump(self.sub_doc_id_to_parent_id, f)

        print("Saving FAISS index...")
        faiss.write_index(self.faiss_index, os.path.join(dirpath, "faiss.index"))

        print(f"Multi-Vector Index saved to {dirpath}")

    def load_index(self, dirpath: str):
        """加载索引数据"""
        parent_content_path = os.path.join(dirpath, "parent_content.pkl")
        mapping_path = os.path.join(dirpath, "sub_to_parent_ids.pkl")
        idx_path = os.path.join(dirpath, "faiss.index")

        if not (os.path.exists(parent_content_path) and os.path.exists(mapping_path) and os.path.exists(idx_path)):
            raise FileNotFoundError(f"Index files not found in {dirpath}")

        print("Loading parent_id_to_content...")
        with open(parent_content_path, "rb") as f:
            self.parent_id_to_content = pickle.load(f)

        print("Loading sub_doc_id_to_parent_id mapping...")
        with open(mapping_path, "rb") as f:
            self.sub_doc_id_to_parent_id = pickle.load(f)

        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index(idx_path)

        self.is_fitted = True
        print(
            f"Index loaded. Parent Docs: {len(self.parent_id_to_content)}, Sub Docs: {len(self.sub_doc_id_to_parent_id)}.")


def main():
    """
    主程序：Multi-Vector 检索器的构建、加载与批量测试。
    """
    data_loader = HQSmallDataLoader(config.BASE_DATA_DIR)
    # 使用较小的切块进行检索
    retriever = MultiVectorRetriever(chunk_size=100, chunk_overlap=20)

    # 1. 索引管理
    index_file_check = os.path.join(config.MULTI_VECTOR_INDEX_DIR, "faiss.index")
    if os.path.exists(index_file_check):
        print(">>> Found existing Multi-Vector index, loading...")
        retriever.load_index(config.MULTI_VECTOR_INDEX_DIR)
    else:
        print(">>> No index found, building from scratch...")
        print("Loading document collection...")
        doc_ids, documents = data_loader.load_collection(config.COLLECTION_PATH)
        retriever.fit(documents, doc_ids)
        retriever.save_index(config.MULTI_VECTOR_INDEX_DIR)

    # 2. 单条查询测试 (Sanity Check)
    sample_query = "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?"
    print(f"\n>>> Sanity Check Query: {sample_query}")
    results = retriever.query(sample_query, top_k=10)
    print("Top 10 results:")
    for i, (doc_id, content, score) in enumerate(results, 1):
        print(f"  {i}. DocID: {doc_id}, Score: {score:.4f}")

    # 3. 批量处理测试集 (Test Set)
    print(f"\n>>> Batch Processing Test Set from {config.VALIDATION_SET_PATH}...")
    test_queries_data = data_loader.load_test_set(config.VALIDATION_SET_PATH)

    batch_results = []
    # 查询部分现在是在循环中单步调用 retriever.query
    for query_item in tqdm(test_queries_data, desc="Retrieving"):
        query_id = query_item["id"]
        query_text = query_item["text"]

        # 检索返回 父文档 ID, 内容 和 分数
        results = retriever.query(query_text, top_k=10)

        batch_results.append({
            "id": query_id,
            "question": query_text,
            # 这里我们只保留 ID 和 Score，与原文件的输出格式一致，方便评估
            "retrieved_docs": [[doc_id, float(score)] for doc_id, content, score in results]
        })

    # 4. 保存
    print(f"Saving predictions to {config.MULTI_VECTOR_OUTPUT_PATH}...")
    with open(config.MULTI_VECTOR_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for result in tqdm(batch_results, desc="Writing file"):
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(">>> Done.")


if __name__ == "__main__":
    main()