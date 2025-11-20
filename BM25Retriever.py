import json
import os
import pickle
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from HQSmallDataLoader import HQSmallDataLoader


class BM25Retriever:
    """
    BM25检索器实现
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        """
        初始化BM25检索器

        Args:
            k1: BM25参数k1
            b: BM25参数b
            epsilon: 平滑参数
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # 文档相关数据结构
        self.documents = []  # 原始文档列表
        self.doc_ids = []  # 文档ID列表
        self.doc_lengths = []  # 文档长度列表
        self.avg_doc_length = 0  # 平均文档长度

        # 词汇表相关
        self.vocab = {}  # 词汇表: term -> index
        self.inverted_index = {}  # 倒排索引: term -> {doc_index -> tf}
        self.idf = {}  # 逆文档频率: term -> idf值

        self.is_fitted = False

    def _preprocess_text(self, text: str) -> List[str]:
        """
        文本预处理（简单的英文分词和小写化）

        Args:
            text: 输入文本

        Returns:
            分词后的token列表
        """
        # 简单的英文分词：转小写，按空格分割，移除标点
        tokens = text.lower().split()
        # 移除标点符号（简单版本）
        tokens = [token.strip('.,!?;:"()[]{}') for token in tokens]
        tokens = [token for token in tokens if token]  # 移除空字符串
        return tokens

    def build_vocab(self, documents: List[str]) -> None:
        """
        构建词汇表和倒排索引

        Args:
            documents: 文档列表
        """
        self.documents = documents
        self.doc_lengths = []

        print("Building vocabulary...")
        # 第一遍：构建词汇表和文档长度
        for doc_idx, doc_text in enumerate(tqdm(documents, desc="Processing documents")):
            tokens = self._preprocess_text(doc_text)
            self.doc_lengths.append(len(tokens))

            # 更新词汇表
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        # 计算平均文档长度
        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 0

        print("Building inverted index...")
        # 第二遍：构建倒排索引
        self.inverted_index = {term: {} for term in self.vocab}

        for doc_idx, doc_text in enumerate(tqdm(documents, desc="Building inverted index")):
            tokens = self._preprocess_text(doc_text)
            term_freq = {}

            # 计算词频
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1

            # 更新倒排索引
            for token, tf in term_freq.items():
                self.inverted_index[token][doc_idx] = tf

    def compute_idf(self) -> None:
        """
        计算逆文档频率(IDF)
        """
        total_docs = len(self.documents)

        print("Computing IDF scores...")
        for term, doc_dict in tqdm(self.inverted_index.items(), desc="Calculating IDF"):
            doc_freq = len(doc_dict)  # 包含该term的文档数量
            # 使用BM25的IDF公式
            self.idf[term] = np.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def fit(self, documents: List[str], doc_ids: List[str] = None) -> None:
        """
        训练BM25模型

        Args:
            documents: 文档列表
            doc_ids: 文档ID列表，如果为None则使用索引作为ID
        """
        if doc_ids is None:
            self.doc_ids = [str(i) for i in range(len(documents))]
        else:
            self.doc_ids = doc_ids

        print("Building vocabulary and inverted index...")
        self.build_vocab(documents)

        print("Computing IDF scores...")
        self.compute_idf()

        self.is_fitted = True
        print(f"BM25 model fitted successfully. Vocabulary size: {len(self.vocab)}, Documents: {len(documents)}")

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        执行查询，返回最相关的文档

        Args:
            query_text: 查询文本
            top_k: 返回的top K结果

        Returns:
            文档ID和得分的列表
        """
        if not self.is_fitted:
            raise ValueError("BM25 model not fitted. Call fit() first.")

        query_tokens = self._preprocess_text(query_text)
        scores = np.zeros(len(self.documents))

        # 计算每个文档的BM25得分
        for token in query_tokens:
            if token not in self.vocab:
                continue

            idf = self.idf[token]
            doc_dict = self.inverted_index[token]

            for doc_idx, tf in doc_dict.items():
                doc_length = self.doc_lengths[doc_idx]
                # BM25公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                scores[doc_idx] += idf * numerator / denominator

        # 获取top K结果
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回得分大于0的结果
                results.append((self.doc_ids[idx], float(scores[idx])))

        return results

    def save_index(self, filepath: str) -> None:
        """
        保存索引到文件

        Args:
            filepath: 文件路径
        """
        if not self.is_fitted:
            raise ValueError("No index to save. Call fit() first.")

        index_data = {
            'documents': self.documents,
            'doc_ids': self.doc_ids,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'vocab': self.vocab,
            'inverted_index': self.inverted_index,
            'idf': self.idf,
            'parameters': {
                'k1': self.k1,
                'b': self.b,
                'epsilon': self.epsilon
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)

        print(f"Index saved to {filepath}")

    def load_index(self, filepath: str) -> None:
        """
        从文件加载索引

        Args:
            filepath: 文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")

        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)

        self.documents = index_data['documents']
        self.doc_ids = index_data['doc_ids']
        self.doc_lengths = index_data['doc_lengths']
        self.avg_doc_length = index_data['avg_doc_length']
        self.vocab = index_data['vocab']
        self.inverted_index = index_data['inverted_index']
        self.idf = index_data['idf']

        # 加载参数
        if 'parameters' in index_data:
            self.k1 = index_data['parameters']['k1']
            self.b = index_data['parameters']['b']
            self.epsilon = index_data['parameters']['epsilon']

        self.is_fitted = True
        print(f"Index loaded from {filepath}. Vocabulary size: {len(self.vocab)}, Documents: {len(self.documents)}")


def main():
    """
    主函数：演示BM25检索器的使用
    """
    # 初始化数据加载器
    data_loader = HQSmallDataLoader("./data")

    # 假设数据文件路径（请根据实际情况修改）
    collection_path = "./COMP5423-25Fall-HQ-small/collection.jsonl"
    train_set_path = "./COMP5423-25Fall-HQ-small/train.jsonl"
    index_save_path = "./data/bm25_index.pkl"

    # 初始化BM25检索器
    retriever = BM25Retriever()

    # 检查是否已有保存的索引
    if os.path.exists(index_save_path):
        print("Loading existing index...")
        retriever.load_index(index_save_path)
    else:
        print("Building new index...")
        # 加载文档集合
        print("Loading document collection...")
        doc_ids, documents = data_loader.load_collection(collection_path)

        # 训练BM25模型
        retriever.fit(documents, doc_ids)

        # 保存索引
        retriever.save_index(index_save_path)

    # 演示查询
    test_queries = [
        "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?"
    ]

    print("\n=== Testing BM25 Retriever ===")
    for query in tqdm(test_queries, desc="Testing queries"):
        print(f"\nQuery: {query}")
        results = retriever.query(query, top_k=5)

        print("Top 5 results:")
        for i, (doc_id, score) in enumerate(results, 1):
            print(f"  {i}. DocID: {doc_id}, Score: {score:.4f}")

    # 演示批量处理（用于生成测试预测）
    print("\n=== Batch Processing Demo ===")
    print("Loading training set...")
    test_queries_data = data_loader.load_train_set(train_set_path)

    batch_results = []
    for query_item in tqdm(test_queries_data, desc="Batch processing queries"):
        query_id = query_item["id"]
        query_text = query_item["text"]

        results = retriever.query(query_text, top_k=10)
        batch_results.append({
            "id": query_id,
            "question": query_text,
            "retrieved_docs": [[doc_id, float(score)] for doc_id, score in results]
        })

    # 保存预测结果（示例）
    output_path = "./data/test_prediction_demo.jsonl"
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in tqdm(batch_results, desc="Writing results"):
            f.write(json.dumps(result) + '\n')

    print(f"Demo predictions saved to {output_path}")


if __name__ == "__main__":
    main()