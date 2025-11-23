import json
import os
import pickle
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

# 导入数据加载器和统一配置
from HQSmallDataLoader import HQSmallDataLoader
import config


class BM25Retriever:
    """
    BM25 (Best Matching 25) 概率检索算法实现。
    基于词频 (TF) 和逆文档频率 (IDF) 进行相关性评分，是稀疏检索的基准算法。
    """

    def __init__(self, k1: float = config.BM25_K1, b: float = config.BM25_B, epsilon: float = config.BM25_EPSILON):
        """
        初始化 BM25 检索器。

        Args:
            k1 (float): 控制词频饱和度的参数 (通常在 1.2 到 2.0 之间)。
            b (float): 控制文档长度归一化的参数 (0 到 1 之间，1 表示完全归一化)。
            epsilon (float): 用于处理负 IDF 值的平滑参数。
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # 文档数据容器
        self.documents = []  # 存储文档原文
        self.doc_ids = []  # 存储文档 ID
        self.doc_lengths = []  # 存储每篇文档的分词长度
        self.avg_doc_length = 0  # 文档库的平均长度

        # 索引结构
        self.vocab = {}  # 词汇表: token -> id
        self.inverted_index = {}  # 倒排索引: token -> {doc_idx: freq}
        self.idf = {}  # 逆文档频率: token -> idf_score

        self.is_fitted = False  # 标记模型是否已训练完毕

    def _preprocess_text(self, text: str) -> List[str]:
        """
        文本预处理流程：
        1. 转小写
        2. 按空格分词
        3. 去除首尾标点符号
        """
        tokens = text.lower().split()
        tokens = [token.strip('.,!?;:"()[]{}') for token in tokens]
        tokens = [token for token in tokens if token]
        return tokens

    def _build_vocab_and_index(self, documents: List[str]) -> None:
        """
        内部方法：遍历文档集，构建词汇表、计算文档长度并建立倒排索引。
        """
        self.documents = documents
        self.doc_lengths = []

        # 步骤 1: 建立词汇表并统计文档长度
        print("Building vocabulary...")
        for doc_text in tqdm(documents, desc="Processing documents (Vocab)"):
            tokens = self._preprocess_text(doc_text)
            self.doc_lengths.append(len(tokens))
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 0

        # 步骤 2: 建立倒排索引 (Inverted Index)
        # 结构: { "apple": {0: 2, 5: 1}, "banana": {1: 1} } 表示 apple 在文档0出现2次，文档5出现1次
        print("Building inverted index...")
        self.inverted_index = {term: {} for term in self.vocab}
        for doc_idx, doc_text in enumerate(tqdm(documents, desc="Building Index")):
            tokens = self._preprocess_text(doc_text)
            # 统计单篇文档内的词频 (Term Frequency)
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            # 更新倒排索引
            for token, tf in term_freq.items():
                self.inverted_index[token][doc_idx] = tf

    def _compute_idf(self) -> None:
        """
        内部方法：基于构建好的倒排索引计算每个词的 IDF 值。
        使用标准 BM25 IDF 公式。
        """
        total_docs = len(self.documents)
        print("Computing IDF scores...")

        for term, doc_dict in tqdm(self.inverted_index.items(), desc="Calculating IDF"):
            doc_freq = len(doc_dict)  # 包含该词的文档数量
            # BM25 IDF 公式
            idf_score = np.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            self.idf[term] = idf_score

    def fit(self, documents: List[str], doc_ids: List[str] = None) -> None:
        """
        核心训练方法：接收文档列表，完成索引构建全流程。

        Args:
            documents (List[str]): 文档文本列表。
            doc_ids (List[str], optional): 对应的文档 ID 列表。
        """
        if doc_ids is None:
            self.doc_ids = [str(i) for i in range(len(documents))]
        else:
            self.doc_ids = doc_ids

        print("Starting BM25 fitting process...")
        self._build_vocab_and_index(documents)
        self._compute_idf()

        self.is_fitted = True
        print(f"BM25 model fitted. Vocab: {len(self.vocab)}, Docs: {len(documents)}")

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        检索方法：给定查询文本，返回最相关的 Top-K 文档。

        Args:
            query_text (str): 查询字符串。
            top_k (int): 返回结果数量。

        Returns:
            List[Tuple[str, float]]: [(doc_id, score), ...]
        """
        if not self.is_fitted:
            raise ValueError("BM25 model not fitted. Call fit() first.")

        query_tokens = self._preprocess_text(query_text)
        # 初始化所有文档得分为 0
        scores = np.zeros(len(self.documents))

        for token in query_tokens:
            if token not in self.vocab:
                continue

            idf = self.idf[token]
            doc_dict = self.inverted_index[token]  # 获取包含该词的文档列表及 TF

            # 向量化计算得分 (针对包含该词的文档)
            for doc_idx, tf in doc_dict.items():
                doc_len = self.doc_lengths[doc_idx]
                # BM25 核心打分公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[doc_idx] += idf * numerator / denominator

        # 获取得分最高的索引
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            # 过滤掉得分为 0 的结果（可选）
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))

        return results

    def save_index(self, filepath: str) -> None:
        """序列化保存索引到磁盘"""
        if not self.is_fitted:
            raise ValueError("No index to save.")

        index_data = {
            'documents': self.documents,
            'doc_ids': self.doc_ids,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'vocab': self.vocab,
            'inverted_index': self.inverted_index,
            'idf': self.idf,
            'parameters': {'k1': self.k1, 'b': self.b, 'epsilon': self.epsilon}
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"Index saved to {filepath}")

    def load_index(self, filepath: str) -> None:
        """从磁盘加载索引"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")

        print(f"Loading index from {filepath}...")
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)

        self.documents = index_data['documents']
        self.doc_ids = index_data['doc_ids']
        self.doc_lengths = index_data['doc_lengths']
        self.avg_doc_length = index_data['avg_doc_length']
        self.vocab = index_data['vocab']
        self.inverted_index = index_data['inverted_index']
        self.idf = index_data['idf']

        if 'parameters' in index_data:
            self.k1 = index_data['parameters']['k1']
            self.b = index_data['parameters']['b']
            self.epsilon = index_data['parameters']['epsilon']

        self.is_fitted = True
        print("Index loaded successfully.")


def main():
    """
    主程序：
    1. 初始化 DataLoader 和 BM25Retriever。
    2. 加载或重新构建索引。
    3. 加载测试集 (test.jsonl)。
    4. 批量检索并保存结果文件。
    """
    data_loader = HQSmallDataLoader(config.BASE_DATA_DIR)
    retriever = BM25Retriever()

    # 1. 索引管理
    if os.path.exists(config.BM25_INDEX_PATH):
        print(">>> Found existing index, loading...")
        retriever.load_index(config.BM25_INDEX_PATH)
    else:
        print(">>> No index found, building from scratch...")
        print(f"Loading collection from {config.COLLECTION_PATH}...")
        doc_ids, documents = data_loader.load_collection(config.COLLECTION_PATH)
        retriever.fit(documents, doc_ids)
        retriever.save_index(config.BM25_INDEX_PATH)

    # 2. 单条查询测试 (Sanity Check)
    sample_query = "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?"
    print(f"\n>>> Sanity Check Query: {sample_query}")
    results = retriever.query(sample_query, top_k=10)
    print("Top 10 results:")
    for i, (doc_id, score) in enumerate(results, 1):
        print(f"  {i}. DocID: {doc_id}, Score: {score:.4f}")

    # 3. 批量处理测试集
    print(f"\n>>> Batch Processing Test Set from {config.TEST_SET_PATH}...")
    test_queries_data = data_loader.load_test_set(config.TEST_SET_PATH)

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
    print(f"Saving predictions to {config.BM25_OUTPUT_PATH}...")
    with open(config.BM25_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for result in tqdm(batch_results, desc="Writing file"):
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(">>> Done.")


if __name__ == "__main__":
    main()