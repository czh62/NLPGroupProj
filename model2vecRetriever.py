import json
import os
import pickle
from typing import List, Tuple

import numpy as np
from model2vec import StaticModel
from tqdm import tqdm

import config
# 导入数据加载器和统一配置
from HQSmallDataLoader import HQSmallDataLoader


class Model2VecRetriever:
    """
    基于 Model2Vec (Static Embeddings) 的稠密检索器实现。
    将文档和查询转换为向量，并使用余弦相似度进行检索。
    """

    def __init__(self, model_name: str = config.MODEL2VEC_MODEL_NAME):
        """
        初始化 Model2Vec 检索器。

        Args:
            model_name (str): HuggingFace 模型名称 (例如 "minishlab/potion-base-8M")。
        """
        print(f"Initializing Model2Vec with model: {model_name}")
        # 加载静态模型
        self.model = StaticModel.from_pretrained(model_name)

        # 数据容器
        self.doc_ids = []  # 存储文档 ID
        self.doc_embeddings = None  # 存储文档向量矩阵 (numpy array)

        self.is_fitted = False  # 标记索引是否已构建

    def _encode_batch(self, texts: List[str], batch_size: int = config.MODEL2VEC_BATCH_SIZE) -> np.ndarray:
        """
        内部方法：批量编码文本为向量。

        Args:
            texts (List[str]): 文本列表。
            batch_size (int): 批处理大小。

        Returns:
            np.ndarray: 形状为 (len(texts), hidden_dim) 的向量矩阵。
        """
        # model2vec 的 encode 方法通常返回列表或 numpy 数组，这里确保转为 numpy
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return np.array(embeddings)

    def fit(self, documents: List[str], doc_ids: List[str] = None) -> None:
        """
        构建索引：将所有文档编码为向量并存储。

        Args:
            documents (List[str]): 文档文本列表。
            doc_ids (List[str], optional): 对应的文档 ID 列表。
        """
        if doc_ids is None:
            self.doc_ids = [str(i) for i in range(len(documents))]
        else:
            self.doc_ids = doc_ids

        print("Starting Model2Vec encoding process...")

        # 编码文档
        # model2vec 非常快，直接对整个列表编码即可
        self.doc_embeddings = self._encode_batch(documents)

        # 归一化向量以进行余弦相似度计算 (L2 Norm)
        # 如果模型输出已经是归一化的，这一步可以跳过，但为了保险起见通常执行
        norm = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        self.doc_embeddings = self.doc_embeddings / (norm + 1e-10)

        self.is_fitted = True
        print(f"Model2Vec index built. Docs: {len(documents)}, Vector Dim: {self.doc_embeddings.shape[1]}")

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        检索方法：给定查询文本，返回最相关的 Top-K 文档。
        使用点积计算余弦相似度（因为向量已归一化）。

        Args:
            query_text (str): 查询字符串。
            top_k (int): 返回结果数量。

        Returns:
            List[Tuple[str, float]]: [(doc_id, score), ...]
        """
        if not self.is_fitted:
            raise ValueError("Index not built. Call fit() or load_index() first.")

        # 1. 编码查询
        # encode 返回的是 batch 的结果，所以取 [0]
        query_vec = self.model.encode([query_text])[0]

        # 2. 归一化查询向量
        norm = np.linalg.norm(query_vec)
        query_vec = query_vec / (norm + 1e-10)

        # 3. 计算相似度 (矩阵乘法: (1, dim) @ (dim, num_docs) -> (1, num_docs))
        # 使用 dot product 计算余弦相似度
        scores = np.dot(self.doc_embeddings, query_vec) * 100

        # 4. 获取 Top-K
        # argsort 返回从小到大的索引，所以取最后 k 个并反转
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((self.doc_ids[idx], float(scores[idx])))

        return results

    def save_index(self, filepath: str) -> None:
        """序列化保存索引到磁盘"""
        if not self.is_fitted:
            raise ValueError("No index to save.")

        index_data = {
            'doc_ids': self.doc_ids,
            'doc_embeddings': self.doc_embeddings
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"Saving index to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        print("Index saved.")

    def load_index(self, filepath: str) -> None:
        """从磁盘加载索引"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")

        print(f"Loading index from {filepath}...")
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)

        self.doc_ids = index_data['doc_ids']
        self.doc_embeddings = index_data['doc_embeddings']

        self.is_fitted = True
        print(f"Index loaded. Shape: {self.doc_embeddings.shape}")


def main():
    """
    主程序：
    1. 初始化 DataLoader 和 Model2VecRetriever。
    2. 加载或重新构建向量索引。
    3. 加载测试集。
    4. 批量检索并保存结果。
    """
    data_loader = HQSmallDataLoader(config.BASE_DATA_DIR)
    retriever = Model2VecRetriever(model_name=config.MODEL2VEC_MODEL_NAME)

    # 1. 索引管理
    if os.path.exists(config.MODEL2VEC_INDEX_PATH):
        print(">>> Found existing index, loading...")
        retriever.load_index(config.MODEL2VEC_INDEX_PATH)
    else:
        print(">>> No index found, building from scratch...")
        print(f"Loading collection from {config.COLLECTION_PATH}...")
        doc_ids, documents = data_loader.load_collection(config.COLLECTION_PATH)
        retriever.fit(documents, doc_ids)
        retriever.save_index(config.MODEL2VEC_INDEX_PATH)

    # 2. 单条查询测试 (Sanity Check)
    sample_query = "Which airport is located in Maine?"
    print(f"\n>>> Sanity Check Query: {sample_query}")
    results = retriever.query(sample_query, top_k=10)
    print("Top 10 results:")
    for i, (doc_id, score) in enumerate(results, 1):
        print(f"  {i}. DocID: {doc_id}, Score: {score:.4f}")

    # 3. 批量处理测试集
    print(f"\n>>> Batch Processing Test Set from {config.VALIDATION_SET_PATH}...")
    test_queries_data = data_loader.load_test_set(config.VALIDATION_SET_PATH)

    batch_results = []
    for query_item in tqdm(test_queries_data, desc="Retrieving"):
        query_id = query_item["id"]
        query_text = query_item["text"]

        # 执行检索
        results = retriever.query(query_text, top_k=10)

        # 格式化输出
        batch_results.append({
            "id": query_id,
            "question": query_text,
            "retrieved_docs": [[doc_id, float(score)] for doc_id, score in results]
        })

    # 4. 保存预测结果
    print(f"Saving predictions to {config.MODEL2VEC_OUTPUT_PATH}...")
    with open(config.MODEL2VEC_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for result in tqdm(batch_results, desc="Writing file"):
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(">>> Done.")


if __name__ == "__main__":
    main()
