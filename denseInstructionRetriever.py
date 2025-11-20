import json
import os
import pickle
from typing import List, Tuple
import faiss
import numpy as np
import requests
from tqdm import tqdm
from HQSmallDataLoader import HQSmallDataLoader

# ==================== 配置区 ====================
API_URL = "https://api.siliconflow.cn/v1/embeddings"
API_KEY = os.environ.get('SILICONFLOW_API_KEY')          # ←←← 请设置你的真实 key
if not API_KEY:
    raise ValueError("请设置环境变量 SILICONFLOW_API_KEY")

# 推荐使用 Qwen3-Embedding-8B（目前 SiliconFlow 已支持）
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"   # 或者 "Qwen/Qwen3-Embedding-0.6B"（更轻量）

BATCH_SIZE = 256          # SiliconFlow Qwen3-Embedding QPS 较高，可适当调大
VECTOR_DIM = 1024        # Qwen3-Embedding-8B 维度是 4096（0.6B 是 1024，根据实际模型调整）
INDEX_DIR = "./data/qwen3_index"
COLLECTION_PATH = "./COMP5423-25Fall-HQ-small/collection.jsonl"

# Qwen3-Embedding 官方推荐的 instruction（必须加，否则效果大幅下降）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DOCUMENT_INSTRUCTION = ""   # 文档侧不需要加 instruction（官方推荐空字符串）

# ================================================

class Qwen3Retriever:
    """
    基于 Qwen3-Embedding（Instruction-tuned） 的 Dense Retriever
    使用 SiliconFlow 在线 embedding 接口 + FAISS 内积检索
    """
    def __init__(self):
        self.doc_ids: List[str] = []
        self.documents: List[str] = []
        self.index: faiss.IndexFlatIP = None
        self.is_fitted = False

    # ----------- 调用 SiliconFlow Embedding API（支持 instruction）-----------
    def _get_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        all_embeddings = []
        instruction = QUERY_INSTRUCTION if is_query else DOCUMENT_INSTRUCTION

        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batch"):
            batch = texts[i:i + BATCH_SIZE]

            # 按官方要求给每条文本加 instruction
            input_texts = [instruction + text for text in batch]

            payload = {
                "model": MODEL_NAME,
                "input": input_texts,
                "encoding_format": "float"
            }
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            try:
                resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(embeddings)
            except Exception as e:
                print(f"Embedding request failed: {e}")
                print("Response:", resp.text if 'resp' in locals() else "None")
                raise

        embeddings_np = np.array(all_embeddings, dtype=np.float32)
        # Qwen3-Embedding 官方推荐 L2 归一化后使用内积（等价于余弦）
        faiss.normalize_L2(embeddings_np)
        return embeddings_np

    # ----------- FIT：构建索引 -----------
    def fit(self, documents: List[str], doc_ids: List[str] = None):
        self.documents = documents
        self.doc_ids = doc_ids if doc_ids else [str(i) for i in range(len(documents))]

        print(f"Total documents: {len(documents)}，开始获取 Qwen3 向量（文档侧）...")
        doc_vectors = self._get_embeddings(documents, is_query=False)

        print("Building FAISS IndexFlatIP...")
        self.index = faiss.IndexFlatIP(VECTOR_DIM)
        self.index.add(doc_vectors)
        self.is_fitted = True
        print("Qwen3 Retriever fitted successfully！")

    # ----------- QUERY -----------
    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.is_fitted:
            raise ValueError("Qwen3Retriever not fitted yet.")

        print(f"Query: {query_text}")
        query_vec = self._get_embeddings([query_text], is_query=True)  # 注意这里加 query instruction

        scores, indices = self.index.search(query_vec, top_k + 5)  # 多取几个防止过滤
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.doc_ids[idx], float(score)))
        return results[:top_k]

    # ----------- 保存索引 -----------
    def save_index(self, dirpath: str = INDEX_DIR):
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, "doc_ids.pkl"), "wb") as f:
            pickle.dump(self.doc_ids, f)
        faiss.write_index(self.index, os.path.join(dirpath, "faiss.index"))
        with open(os.path.join(dirpath, "documents.jsonl"), "w", encoding="utf-8") as f:
            for doc_id, text in zip(self.doc_ids, self.documents):
                f.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
        print(f"Index saved to {dirpath}")

    # ----------- 加载索引 -----------
    def load_index(self, dirpath: str = INDEX_DIR):
        with open(os.path.join(dirpath, "doc_ids.pkl"), "rb") as f:
            self.doc_ids = pickle.load(f)
        self.index = faiss.read_index(os.path.join(dirpath, "faiss.index"))
        self.is_fitted = True
        print(f"Index loaded from {dirpath}，documents count: {len(self.doc_ids)}")


def main():
    data_loader = HQSmallDataLoader("./data")
    index_dir = INDEX_DIR
    retriever = Qwen3Retriever()

    if os.path.exists(f"{index_dir}/faiss.index"):
        print("Found existing Qwen3 index，loading...")
        retriever.load_index(index_dir)
    else:
        print("No index found，building new Qwen3 index...")
        doc_ids, docs = data_loader.load_collection(COLLECTION_PATH)
        retriever.fit(docs, doc_ids)
        retriever.save_index(index_dir)

    # 测试查询（中英文都支持）
    test_queries = [
        "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?"
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"Query: {q}")
        results = retriever.query(q, top_k=5)
        print("Top-5 Results:")
        for rank, (doc_id, score) in enumerate(results, 1):
            # 可选：打印文档内容便于调试
            # doc_text = next((d for d in retriever.documents if retriever.doc_ids[retriever.documents.index(d)] == doc_id), "")
            print(f" {rank:>2}. [DocID: {doc_id}] Score: {score:.4f}")

if __name__ == "__main__":
    main()