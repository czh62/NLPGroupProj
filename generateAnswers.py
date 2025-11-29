import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import config
from BGEReranker import BGEReranker
from BM25Retriever import BM25Retriever
from HQSmallDataLoader import HQSmallDataLoader
from answerPipeline import call_llm
from denseRetriever import BGERetriever
from hybridRetrieveRerank import hybrid_retrieve_and_rerank
from prompts import GENERATE_ANSWER_PROMPT


def main_hybrid_with_answers():
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
        bge_retriever = BGERetriever(api_key=config.SF_API_KEY)
    else:
        print(f">>> Using Local Ollama")
        bge_reranker = BGEReranker()  # 第一次运行会自动下载模型
        bm25_retriever = BM25Retriever()
        bge_retriever = BGERetriever()

    # 加载索引 (假设你已经按照之前的脚本生成了索引)
    print("Loading indices...")
    bm25_retriever.load_index(config.BM25_INDEX_PATH)
    bge_retriever.load_index(config.BGE_INDEX_DIR)

    sample_query = "Were both Gabriela Mistral and G. K. Chesterton authors?"
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
    doc_texts = []
    for i, (doc_id, score) in enumerate(results, 1):
        print(f"  {i}. DocID: {doc_id}, Score: {score:.4f}")
        if doc_id in doc_id_to_text:
            doc_texts.append(doc_id_to_text[doc_id])
    # 直接生成答案，不进行相关性检查
    context = "\n\n".join(doc_texts)
    gen_prompt = GENERATE_ANSWER_PROMPT.format(
        query=sample_query,
        previous_answers="",
        context=context
    )
    answer = call_llm(gen_prompt, step_name="FALLBACK ANSWER GENERATION")
    print("answer:" + answer)

    # 2. 加载测试集
    test_queries_data = data_loader.load_test_set(config.TEST_SET_PATH)

    # 3. 批量处理
    print(">>> Starting Hybrid Retrieval + Reranking + Answer Generation with X threads...")

    batch_results = []
    previous_answers = {}  # 用于存储前序问题的答案

    def process_one_query(query_item):
        """单个 query 的处理逻辑，用于在线程里执行"""
        query_id = query_item["id"]
        query_text = query_item["text"]

        # 检索文档
        final_top_docs = hybrid_retrieve_and_rerank(
            query=query_text,
            first_retriever=bm25_retriever,
            second_retriever=bge_retriever,
            reranker=bge_reranker,
            doc_id_to_text_map=doc_id_to_text,
            retrieval_top_k=50,
            rerank_top_k=10
        )

        # 获取检索到的文档内容
        retrieved_doc_ids = [doc_id for doc_id, score in final_top_docs]
        doc_texts = []
        for doc_id in retrieved_doc_ids:
            if doc_id in doc_id_to_text:
                doc_texts.append(doc_id_to_text[doc_id])
            else:
                print(f"Warning: Document {doc_id} not found in mapping")

        # 直接生成答案，不进行相关性检查
        context = "\n\n".join(doc_texts)
        gen_prompt = GENERATE_ANSWER_PROMPT.format(
            query=query_text,
            previous_answers="",
            context=context
        )
        answer = call_llm(gen_prompt, step_name="FALLBACK ANSWER GENERATION")

        # 更新前序答案（用于后续问题）
        previous_answers[query_id] = answer

        return {
            "id": query_id,
            "text": query_text,
            "answer": answer,
            "retrieved_docs": [[doc_id, float(score)] for doc_id, score in final_top_docs]
        }

    # 使用 X 线程并发执行全部 query
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_one_query, item) for item in test_queries_data]

        # tqdm 进度条 + 多线程结果收集
        for future in tqdm(as_completed(futures), total=len(futures), desc="Hybrid Pipeline"):
            batch_results.append(future.result())

    # 4. 保存结果
    print(f"Saving hybrid results with answers to {config.FINAL_ANSWERS_PATH}...")
    with open(config.FINAL_ANSWERS_PATH, 'w', encoding='utf-8') as f:
        for result in batch_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(">>> Hybrid Pipeline with Answer Generation Done.")


if __name__ == "__main__":
    main_hybrid_with_answers()
