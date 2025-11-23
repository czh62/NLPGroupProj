import os

from numpy.random import SFC64

# ==================== 路径配置 ====================
# 基础数据目录
BASE_DATA_DIR = "./data"
# 原始数据集目录
RAW_DATA_DIR = "./COMP5423-25Fall-HQ-small"

# 数据集文件路径
COLLECTION_PATH = os.path.join(RAW_DATA_DIR, "collection.jsonl")
VALIDATION_SET_PATH = os.path.join(RAW_DATA_DIR, "validation.jsonl")
TEST_SET_PATH = os.path.join(RAW_DATA_DIR, "test.jsonl")

# ==================== 模型保存路径 ====================
# BM25
BM25_INDEX_PATH = os.path.join(BASE_DATA_DIR, "bm25_index.pkl")
BM25_OUTPUT_PATH = os.path.join(BASE_DATA_DIR, "bm25_test_prediction.jsonl")

# Model2Vec
MODEL2VEC_INDEX_PATH = os.path.join(BASE_DATA_DIR, "model2vec_index.pkl")
MODEL2VEC_OUTPUT_PATH = os.path.join(BASE_DATA_DIR, "model2vec_test_prediction.jsonl")

# BGE-M3
BGE_INDEX_DIR = os.path.join(BASE_DATA_DIR, "bge_index")
BGE_OUTPUT_PATH = os.path.join(BASE_DATA_DIR, "bge_test_prediction.jsonl")

# Qwen3
QWEN_INDEX_DIR = os.path.join(BASE_DATA_DIR, "qwen3_index")
QWEN_OUTPUT_PATH = os.path.join(BASE_DATA_DIR, "qwen3_test_prediction.jsonl")

MULTI_VECTOR_INDEX_DIR = os.path.join(BASE_DATA_DIR, "multi_vector_index")
MULTI_VECTOR_OUTPUT_PATH = os.path.join(BASE_DATA_DIR, "multi_vector_results.jsonl")

# ==================== 算法参数配置 ====================
# BM25 参数
BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25

# Model2Vec 配置
MODEL2VEC_MODEL_NAME = "minishlab/potion-base-32M"
MODEL2VEC_BATCH_SIZE = 256  # 批量编码的大小

# Ollama 服务配置 (用于 BGE 和 Qwen)
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
SF_API_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
SF_API_EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"
SF_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
BATCH_SIZE = 32

# BGE 模型配置
BGE_MODEL_NAME = "bge-m3"
SF_BGE_MODEL_NAME = "BAAI/bge-m3"
BGE_VECTOR_DIM = 1024
BGE_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Qwen3 模型配置
QWEN_MODEL_NAME = "qwen3-embedding:8b"
SF_QWEN_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
QWEN_VECTOR_DIM = 4096
QWEN_QUERY_INSTRUCTION = "Generate a vector representation for this query to retrieve relevant documents: "
QWEN_DOC_INSTRUCTION = ""
