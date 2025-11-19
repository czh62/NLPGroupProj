import json
import os
import pickle
from typing import List, Tuple

import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn.functional as F

from HQSmallDataLoader import HQSmallDataLoader


class Model2VecSimCSERetriever:
    """
    基于model2vec静态词袋 + SimCSE的混合检索器
    """

    def __init__(self,
                 model2vec_weight: float = 0.3,
                 simcse_weight: float = 0.7,
                 simcse_model_name: str = "princeton-nlp/sup-simcse-bert-base-uncased"):
        """
        初始化混合检索器

        Args:
            model2vec_weight: model2vec得分权重
            simcse_weight: SimCSE得分权重
            simcse_model_name: SimCSE模型名称
        """
        self.model2vec_weight = model2vec_weight
        self.simcse_weight = simcse_weight

        # model2vec相关参数
        self.documents = []
        self.doc_ids = []
        self.vocab = {}
        self.inverted_index = {}
        self.idf = {}
        self.doc_vectors = {}  # 文档的model2vec向量

        # SimCSE相关
        self.simcse_model_name = simcse_model_name
        self.simcse_embeddings = None  # 文档的SimCSE嵌入

        self.is_fitted = False

        # 初始化SimCSE模型（延迟加载）
        self.simcse_model = None
        self.simcse_tokenizer = None

    def _load_simcse_model(self):
        """延迟加载SimCSE模型"""
        if self.simcse_model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                print("Loading SimCSE model...")
                self.simcse_tokenizer = AutoTokenizer.from_pretrained(self.simcse_model_name)
                self.simcse_model = AutoModel.from_pretrained(self.simcse_model_name)
                # 设置为评估模式
                self.simcse_model.eval()
                print(f"SimCSE model loaded: {self.simcse_model_name}")
            except ImportError:
                raise ImportError("Please install transformers: pip install transformers")
            except Exception as e:
                raise RuntimeError(f"Failed to load SimCSE model: {e}")

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

    def _get_simcse_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的SimCSE嵌入

        Args:
            text: 输入文本

        Returns:
            文本嵌入向量
        """
        self._load_simcse_model()

        # Tokenize
        inputs = self.simcse_tokenizer(text,
                                       return_tensors="pt",
                                       truncation=True,
                                       max_length=512,
                                       padding=True)

        # 生成嵌入
        with torch.no_grad():
            outputs = self.simcse_model(**inputs)
            # 使用[CLS] token的嵌入作为句子表示
            embeddings = outputs.last_hidden_state[:, 0, :]
            # 归一化
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings[0].numpy()

    def _compute_model2vec_vector(self, tokens: List[str]) -> np.ndarray:
        """
        计算model2vec向量（基于TF-IDF的静态词袋）

        Args:
            tokens: 分词后的token列表

        Returns:
            model2vec向量
        """
        vector = np.zeros(len(self.vocab))

        # 计算TF
        term_freq = {}
        for token in tokens:
            if token in self.vocab:
                term_freq[token] = term_freq.get(token, 0) + 1

        # 构建TF-IDF向量
        for token, tf in term_freq.items():
            if token in self.vocab and token in self.idf:
                idx = self.vocab[token]
                vector[idx] = tf * self.idf[token]

        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def build_vocab_and_index(self, documents: List[str]) -> None:
        """
        构建词汇表、倒排索引和计算IDF

        Args:
            documents: 文档列表
        """
        self.documents = documents
        self.vocab = {}
        self.inverted_index = {}

        # 第一遍：构建词汇表
        print("Building vocabulary...")
        for doc_idx, doc_text in enumerate(documents):
            tokens = self._preprocess_text(doc_text)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        # 第二遍：构建倒排索引
        print("Building inverted index...")
        self.inverted_index = {term: {} for term in self.vocab}

        for doc_idx, doc_text in enumerate(documents):
            tokens = self._preprocess_text(doc_text)
            term_freq = {}

            for token in tokens:
                if token in self.vocab:
                    term_freq[token] = term_freq.get(token, 0) + 1

            for token, tf in term_freq.items():
                self.inverted_index[token][doc_idx] = tf

    def compute_idf(self) -> None:
        """
        计算逆文档频率(IDF)
        """
        total_docs = len(self.documents)
        self.idf = {}

        for term, doc_dict in self.inverted_index.items():
            doc_freq = len(doc_dict)
            # 标准IDF公式
            self.idf[term] = np.log((total_docs + 1) / (doc_freq + 1)) + 1

    def compute_model2vec_vectors(self) -> None:
        """
        计算所有文档的model2vec向量
        """
        print("Computing model2vec vectors...")
        self.doc_vectors = {}

        for doc_idx, doc_text in enumerate(self.documents):
            tokens = self._preprocess_text(doc_text)
            vector = self._compute_model2vec_vector(tokens)
            self.doc_vectors[doc_idx] = vector

    def compute_simcse_embeddings(self) -> None:
        """
        计算所有文档的SimCSE嵌入
        """
        print("Computing SimCSE embeddings...")
        self.simcse_embeddings = {}

        for doc_idx, doc_text in enumerate(self.documents):
            if doc_idx % 1000 == 0:
                print(f"Processing document {doc_idx}/{len(self.documents)}")
            embedding = self._get_simcse_embedding(doc_text)
            self.simcse_embeddings[doc_idx] = embedding

    def fit(self, documents: List[str], doc_ids: List[str] = None) -> None:
        """
        训练混合检索模型

        Args:
            documents: 文档列表
            doc_ids: 文档ID列表，如果为None则使用索引作为ID
        """
        if doc_ids is None:
            self.doc_ids = [str(i) for i in range(len(documents))]
        else:
            self.doc_ids = doc_ids

        # 构建model2vec部分
        print("Fitting model2vec component...")
        self.build_vocab_and_index(documents)
        self.compute_idf()
        self.compute_model2vec_vectors()

        # 构建SimCSE部分
        print("Fitting SimCSE component...")
        self.compute_simcse_embeddings()

        self.is_fitted = True
        print(f"Hybrid model fitted successfully. Vocabulary size: {len(self.vocab)}, Documents: {len(documents)}")

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        执行混合查询，返回最相关的文档

        Args:
            query_text: 查询文本
            top_k: 返回的top K结果

        Returns:
            文档ID和得分的列表
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # 计算查询的model2vec向量
        query_tokens = self._preprocess_text(query_text)
        query_model2vec = self._compute_model2vec_vector(query_tokens)

        # 计算查询的SimCSE嵌入
        query_simcse = self._get_simcse_embedding(query_text)

        scores = {}

        # 计算每个文档的混合得分
        for doc_idx in range(len(self.documents)):
            # model2vec相似度（余弦相似度）
            if doc_idx in self.doc_vectors:
                model2vec_sim = np.dot(query_model2vec, self.doc_vectors[doc_idx])
            else:
                model2vec_sim = 0

            # SimCSE相似度（余弦相似度）
            if doc_idx in self.simcse_embeddings:
                simcse_sim = np.dot(query_simcse, self.simcse_embeddings[doc_idx])
            else:
                simcse_sim = 0

            # 混合得分
            hybrid_score = (self.model2vec_weight * model2vec_sim +
                            self.simcse_weight * simcse_sim)

            scores[doc_idx] = hybrid_score

        # 获取top K结果
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_idx, score in sorted_docs:
            if score > 0:  # 只返回得分大于0的结果
                results.append((self.doc_ids[doc_idx], float(score)))

        return results

    def save_index(self, filepath: str) -> None:
        """
        保存索引到文件

        Args:
            filepath: 文件路径
        """
        if not self.is_fitted:
            raise ValueError("No index to save. Call fit() first.")

        # 转换numpy数组为列表以便序列化
        doc_vectors_serializable = {k: v.tolist() for k, v in self.doc_vectors.items()}
        simcse_embeddings_serializable = {k: v.tolist() for k, v in self.simcse_embeddings.items()}

        index_data = {
            'documents': self.documents,
            'doc_ids': self.doc_ids,
            'vocab': self.vocab,
            'inverted_index': self.inverted_index,
            'idf': self.idf,
            'doc_vectors': doc_vectors_serializable,
            'simcse_embeddings': simcse_embeddings_serializable,
            'parameters': {
                'model2vec_weight': self.model2vec_weight,
                'simcse_weight': self.simcse_weight,
                'simcse_model_name': self.simcse_model_name
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
        self.vocab = index_data['vocab']
        self.inverted_index = index_data['inverted_index']
        self.idf = index_data['idf']

        # 转换回numpy数组
        self.doc_vectors = {k: np.array(v) for k, v in index_data['doc_vectors'].items()}
        self.simcse_embeddings = {k: np.array(v) for k, v in index_data['simcse_embeddings'].items()}

        # 加载参数
        if 'parameters' in index_data:
            self.model2vec_weight = index_data['parameters']['model2vec_weight']
            self.simcse_weight = index_data['parameters']['simcse_weight']
            self.simcse_model_name = index_data['parameters']['simcse_model_name']

        self.is_fitted = True
        print(f"Index loaded from {filepath}. Vocabulary size: {len(self.vocab)}, Documents: {len(self.documents)}")


def main():
    """
    主函数：演示混合检索器的使用
    """
    from tqdm import tqdm

    # 初始化数据加载器
    data_loader = HQSmallDataLoader("./data")

    # 假设数据文件路径（请根据实际情况修改）
    collection_path = "./COMP5423-25Fall-HQ-small/collection.jsonl"
    train_set_path = "./COMP5423-25Fall-HQ-small/train.jsonl"
    index_save_path = "./data/model2vec_simcse_index.pkl"

    # 初始化混合检索器
    retriever = Model2VecSimCSERetriever()

    # 检查是否已有保存的索引
    if os.path.exists(index_save_path):
        print("Loading existing index...")
        retriever.load_index(index_save_path)
    else:
        print("Building new index...")
        # 加载文档集合
        doc_ids, documents = data_loader.load_collection(collection_path)
        # 训练混合模型
        retriever.fit(documents, doc_ids)

        # 保存索引
        retriever.save_index(index_save_path)

    # 演示查询
    test_queries = [
        "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?",
        "What is the capital of France?",
        "Who wrote the novel Pride and Prejudice?"
    ]

    print("\n=== Testing Model2Vec+SimCSE Retriever ===")
    for query in tqdm(test_queries, desc="Testing queries"):
        print(f"\nQuery: {query}")
        results = retriever.query(query, top_k=5)

        print("Top 5 results:")
        for i, (doc_id, score) in enumerate(results, 1):
            print(f"  {i}. DocID: {doc_id}, Score: {score:.4f}")

    # 演示批量处理（用于生成测试预测）
    print("\n=== Batch Processing Demo ===")
    test_queries_data = data_loader.load_train_set(train_set_path)

    # 为了演示，使用前50个查询
    test_queries_data = test_queries_data[:50]

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
    output_path = "./data/model2vec_simcse_predictions.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in tqdm(batch_results, desc="Saving results"):
            f.write(json.dumps(result) + '\n')

    print(f"Demo predictions saved to {output_path}")


if __name__ == "__main__":
    main()
