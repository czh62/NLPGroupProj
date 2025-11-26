import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from tqdm import tqdm

import config
from BGEReranker import BGEReranker
from BM25Retriever import BM25Retriever
from HQSmallDataLoader import HQSmallDataLoader
from denseInstructionRetriever import Qwen3Retriever
from denseRetriever import BGERetriever


def reciprocal_rank_fusion(results_list: List[List[Tuple[str, float]]], k: int = 60):
    """
    倒数排名融合 (RRF) 算法。

    Args:
        results_list: 包含多个检索结果列表的列表，每个结果列表为 [(doc_id, score), ...]
        k: 平滑常数，通常设为 60
    """
    fused_scores = {}

    for results in results_list:
        for rank, (doc_id, _) in enumerate(results):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            # RRF 公式: score += 1 / (k + rank)
            fused_scores[doc_id] += 1.0 / (k + rank)

    # 转换为列表并排序
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def hybrid_retrieve_and_rerank(
        query: str,
        first_retriever,
        second_retriever,
        reranker,
        doc_id_to_text_map: dict,
        retrieval_top_k: int = 50,
        rerank_top_k: int = 10
):
    """
    执行混合检索 + 重排序流程。

    流程:
    1. BM25 检索 Top-K
    2. BGE 检索 Top-K
    3. RRF 融合并去重，取前 N 个候选
    4. Reranker 对候选进行精排
    5. 返回最终 Top-K
    """

    # 1. 并行检索 (此处为串行调用，实际生产可改为多线程)
    first_results = first_retriever.query(query, top_k=retrieval_top_k)
    second_results = second_retriever.query(query, top_k=retrieval_top_k)

    # 2. 融合结果 (RRF)
    fused_results = reciprocal_rank_fusion([first_results, second_results])

    # 3. 准备重排序候选集 (Candidate Generation)
    # 我们取出 RRF 排名靠前的文档送入 Reranker
    # 注意：Reranker 很慢，所以candidate_k 不要太大 (通常 50-100)
    candidate_k = min(len(fused_results), int(retrieval_top_k * 1.5))
    candidates = fused_results[:candidate_k]

    candidate_ids = [doc_id for doc_id, _ in candidates]
    candidate_texts = []

    # 获取文档文本用于重排序
    for doc_id in candidate_ids:
        # 这里假设你有一个 {id: text} 的字典映射，这是最高效的方式
        text = doc_id_to_text_map.get(doc_id, "")
        candidate_texts.append(text)

    # 4. 执行重排序 (Reranking)
    # 返回格式: [(doc_id, rerank_score, original_idx), ...]
    reranked_results = reranker.rerank(query, candidate_texts, candidate_ids)

    # 5. 截取最终所需的 Top-K
    final_results = reranked_results[:rerank_top_k]

    # 格式化输出 [(doc_id, score), ...]
    return [(res[0], res[1]) for res in final_results]


def main_hybrid():
    # 1. 初始化数据和检索器
    data_loader = HQSmallDataLoader(config.BASE_DATA_DIR)

    # 加载原始数据构建 ID->Text 映射 (为了 Reranker 取文本)
    print("Loading collection for lookup map...")
    all_doc_ids, all_documents = data_loader.load_collection(config.COLLECTION_PATH)
    doc_id_to_text = dict(zip(all_doc_ids, all_documents))

    if getattr(config, "SF_API_KEY", None):
        print(f">>> Using SiliconFlow API")
        bge_reranker = BGEReranker(api_key=config.SF_API_KEY)
        bm25_retriever = BM25Retriever()
        qwen3_retriever = Qwen3Retriever(api_key=config.SF_API_KEY)
        bge_retriever = BGERetriever(api_key=config.SF_API_KEY)
    else:
        print(f">>> Using Local Ollama")
        bge_reranker = BGEReranker()  # 第一次运行会自动下载模型
        bm25_retriever = BM25Retriever()
        qwen3_retriever = Qwen3Retriever()
        bge_retriever = BGERetriever()

    # 加载索引 (假设你已经按照之前的脚本生成了索引)
    print("Loading indices...")
    bm25_retriever.load_index(config.BM25_INDEX_PATH)
    bge_retriever.load_index(config.BGE_INDEX_DIR)
    qwen3_retriever.load_index(config.QWEN_INDEX_DIR)

    # 2. 加载测试集
    test_queries_data = data_loader.load_test_set(config.VALIDATION_SET_PATH)

    sample_query = "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?"
    print(f"\n>>> Sanity Check Query: {sample_query}")
    results = hybrid_retrieve_and_rerank(query=sample_query,
                                         first_retriever=bm25_retriever,
                                         second_retriever=bge_retriever,
                                         reranker=bge_reranker,
                                         doc_id_to_text_map=doc_id_to_text,
                                         retrieval_top_k=50,  # 粗排召回数量
                                         rerank_top_k=10  # 最终输出数量
                                         )
    print("Top 10 results:")
    for i, (doc_id, score) in enumerate(results, 1):
        print(f"  {i}. DocID: {doc_id}, Score: {score:.4f}")

    # 3. 批量处理
    print(">>> Starting Hybrid Retrieval + Reranking with X threads...")

    batch_results = []

    def process_one_query(query_item):
        """单个 query 的处理逻辑，用于在线程里执行"""
        query_id = query_item["id"]
        query_text = query_item["text"]

        final_top_docs = hybrid_retrieve_and_rerank(
            query=query_text,
            first_retriever=bm25_retriever,
            second_retriever=bge_retriever,
            reranker=bge_reranker,
            doc_id_to_text_map=doc_id_to_text,
            retrieval_top_k=50,
            rerank_top_k=10
        )

        return {
            "id": query_id,
            "question": query_text,
            "retrieved_docs": [[doc_id, float(score)] for doc_id, score in final_top_docs]
        }

    # 使用 X 线程并发执行全部 query
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_one_query, item) for item in test_queries_data]

        # tqdm 进度条 + 多线程结果收集
        for future in tqdm(as_completed(futures), total=len(futures), desc="Hybrid Pipeline"):
            batch_results.append(future.result())

    # 4. 保存结果
    print(f"Saving hybrid results to {config.HYBRID_RETRIEVE_AND_RERANK_OUTPUT_PATH}...")
    with open(config.HYBRID_RETRIEVE_AND_RERANK_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for result in batch_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(">>> Hybrid Pipeline Done.")


if __name__ == "__main__":
    main_hybrid()
