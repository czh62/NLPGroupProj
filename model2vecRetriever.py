import math
import os
import pickle
from collections import Counter
from typing import List, Tuple

import faiss
import nltk
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from tqdm import tqdm

from HQSmallDataLoader import HQSmallDataLoader

# 确保第一次运行时下载 stopwords
try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


class Word2VecRetriever:
    """
    使用 Word2Vec + TF-IDF 加权向量 + FAISS 的增强检索器
    """

    def __init__(self, vector_dim: int = 200, window: int = 5, min_count: int = 1):
        self.vector_dim = vector_dim
        self.window = window
        self.min_count = min_count

        self.doc_ids: List[str] = []
        self.documents: List[str] = []

        self.w2v_model: Word2Vec = None

        self.index: faiss.IndexFlatIP = None
        self.doc_vectors: np.ndarray = None

        self.term_df = None
        self.is_fitted = False

    # ----------- 文本预处理（增强：停用词过滤） -----------
    def _preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        tokens = text.split()
        tokens = [t.strip(".,!?;:\"()[]{}") for t in tokens]
        tokens = [t for t in tokens if t and t not in stop_words]
        return tokens

    # ----------- Word2Vec 训练 -----------
    def _train_word2vec(self, tokenized_docs: List[List[str]]):
        print("Training Word2Vec...")
        self.w2v_model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.vector_dim,
            window=self.window,
            min_count=self.min_count,
            workers=4
        )
        print("Word2Vec training done.")

    # ----------- TF-IDF + Word2Vec 文档向量计算（核心增强） -----------
    def _get_doc_vector(self, tokens: List[str]) -> np.ndarray:
        vectors = []
        token_count = Counter(tokens)

        total_docs = len(self.documents) + 1  # 避免除 0

        for t in tokens:
            if t in self.w2v_model.wv:
                # TF
                tf = token_count[t] / len(tokens)

                # IDF：df + 1 平滑处理
                df = self.term_df.get(t, 0) + 1
                idf = math.log(total_docs / df)

                weight = tf * idf
                vectors.append(weight * self.w2v_model.wv[t])

        if not vectors:
            return np.zeros(self.vector_dim, dtype=np.float32)

        return np.mean(vectors, axis=0).astype(np.float32)

    # ----------- 构建所有文档向量 -----------
    def _build_doc_vectors(self, tokenized_docs: List[List[str]]):
        print("Building weighted document vectors...")
        vectors = []
        for tokens in tqdm(tokenized_docs):
            vectors.append(self._get_doc_vector(tokens))

        self.doc_vectors = np.vstack(vectors)
        faiss.normalize_L2(self.doc_vectors)

    # ----------- FIT -----------
    def fit(self, documents: List[str], doc_ids: List[str] = None):
        self.documents = documents
        self.doc_ids = doc_ids if doc_ids else [str(i) for i in range(len(documents))]

        print("Preprocessing documents...")
        tokenized_docs = [self._preprocess_text(doc) for doc in documents]

        # 统计 document frequency
        print("Counting document frequency (DF)...")
        df_counter = Counter()
        for tokens in tokenized_docs:
            for t in set(tokens):
                df_counter[t] += 1
        self.term_df = df_counter

        # 训练 Word2Vec
        self._train_word2vec(tokenized_docs)

        # 文档向量构建
        self._build_doc_vectors(tokenized_docs)

        # 构建 FAISS 索引
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.vector_dim)
        self.index.add(self.doc_vectors)

        self.is_fitted = True
        print(f"Word2Vec retriever fitted. Documents: {len(documents)}")

    # ----------- QUERY -----------
    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.is_fitted:
            raise ValueError("Word2VecRetriever not fitted.")

        tokens = self._preprocess_text(query_text)
        query_vec = self._get_doc_vector(tokens).reshape(1, -1)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        result = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            result.append((self.doc_ids[idx], float(score)))

        return result

    # ----------- 保存 -----------
    def save_index(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)

        print("Saving Word2Vec model...")
        self.w2v_model.save(os.path.join(dirpath, "word2vec.model"))

        print("Saving doc ids...")
        with open(os.path.join(dirpath, "doc_ids.pkl"), "wb") as f:
            pickle.dump(self.doc_ids, f)

        print("Saving document frequency info...")
        with open(os.path.join(dirpath, "term_df.pkl"), "wb") as f:
            pickle.dump(self.term_df, f)

        print("Saving FAISS index...")
        faiss.write_index(self.index, os.path.join(dirpath, "faiss.index"))

        print("Index saved.")

    # ----------- 加载 -----------
    def load_index(self, dirpath: str):
        print("Loading Word2Vec model...")
        self.w2v_model = Word2Vec.load(os.path.join(dirpath, "word2vec.model"))

        print("Loading doc ids...")
        with open(os.path.join(dirpath, "doc_ids.pkl"), "rb") as f:
            self.doc_ids = pickle.load(f)

        print("Loading DF info...")
        with open(os.path.join(dirpath, "term_df.pkl"), "rb") as f:
            self.term_df = pickle.load(f)

        print("Loading FAISS index...")
        self.index = faiss.read_index(os.path.join(dirpath, "faiss.index"))

        self.is_fitted = True
        print("Index loaded successfully.")


def main():
    data_loader = HQSmallDataLoader("./data")

    collection_path = "./COMP5423-25Fall-HQ-small/collection.jsonl"
    index_dir = "./data/w2v_index"

    retriever = Word2VecRetriever(vector_dim=200)

    if os.path.exists(f"{index_dir}/faiss.index"):
        print("Loading existing index...")
        retriever.load_index(index_dir)
    else:
        print("Building new index...")
        doc_ids, documents = data_loader.load_collection(collection_path)

        retriever.fit(documents, doc_ids)
        retriever.save_index(index_dir)

    test_queries = [
        "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?"
    ]
    for query in tqdm(test_queries, desc="Testing queries"):
        print(f"\nQuery: {query}")
        results = retriever.query(query, top_k=5)

        print("Results:")
        for i, (doc_id, score) in enumerate(results, 1):
            print(f"  {i}. DocID: {doc_id}, Score: {score:.4f}")


if __name__ == "__main__":
    main()
