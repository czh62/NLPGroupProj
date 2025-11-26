import json
import requests
import re
import config  # 假设 config.py 存在，包含 BASE_DATA_DIR, SF_API_KEY 等配置
from BGEReranker import BGEReranker  # BGE 重排序器模块
from BM25Retriever import BM25Retriever  # BM25 检索器模块
from HQSmallDataLoader import HQSmallDataLoader  # HotpotQA 小型数据集加载器
from denseInstructionRetriever import Qwen3Retriever  # Qwen3 密集检索器
from denseRetriever import BGERetriever  # BGE 密集检索器
from hybridRetrieveRerank import hybrid_retrieve_and_rerank  # 混合检索和重排序函数
from prompts import DECOMPOSITION_PROMPT, RELEVANCE_CHECK_PROMPT, QUERY_REWRITE_PROMPT, GENERATE_ANSWER_PROMPT, \
    SELF_CHECK_PROMPT, SYNTHESIZE_ANSWERS_PROMPT  # 导入预定义的提示模板


# 函数：清理和解析 JSON 响应
def clean_and_parse_json_response(response_text, step_name=""):
    """
    清理响应文本中的 JSON 标记并解析为字典。
    参数:
    response_text (str): 包含 JSON 的响应文本
    step_name (str): 步骤名称，用于日志记录
    返回:
    dict: 解析后的 JSON 字典
    """
    print(f"🧹 Cleaning JSON response for {step_name}...")
    print(f"📥 Raw response: {response_text}")

    # 尝试直接解析
    try:
        result = json.loads(response_text)
        print(f"   ✅ Direct JSON parsing successful")
        return result
    except json.JSONDecodeError:
        print(f"   ⚠️ Direct parsing failed, attempting to extract JSON from markdown code blocks")
        pass

    # 清理常见的 JSON 标记
    cleaned_text = response_text.strip()

    # 移除 ```json 和 ``` 标记
    cleaned_text = re.sub(r'^```json\s*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'```\s*$', '', cleaned_text)

    # 移除其他可能的代码块标记
    cleaned_text = re.sub(r'^```\s*', '', cleaned_text)
    cleaned_text = re.sub(r'```\s*$', '', cleaned_text)

    # 移除开头的 "json" 字样
    cleaned_text = re.sub(r'^json\s*', '', cleaned_text, flags=re.IGNORECASE)

    cleaned_text = cleaned_text.strip()
    print(f"   🔧 Cleaned text: {cleaned_text}")

    # 尝试解析清理后的文本
    try:
        result = json.loads(cleaned_text)
        print(f"   ✅ Cleaned JSON parsing successful")
        return result
    except json.JSONDecodeError as e:
        print(f"   ❌ Failed to parse JSON after cleaning: {e}")
        # 如果还是失败，尝试更宽松的提取方法
        try:
            # 查找第一个 { 和最后一个 }
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = cleaned_text[start_idx:end_idx]
                result = json.loads(json_str)
                print(f"   ✅ Extracted JSON parsing successful")
                return result
        except Exception as e2:
            print(f"   ❌ All JSON parsing attempts failed: {e2}")

    # 如果所有解析都失败，返回默认结构
    default_result = {
        "is_relevant": False,
        "reason": "JSON parsing failed",
        "suggested_rewrite": ""
    }
    print(f"   ⚠️ Returning default result due to parsing failure")
    return default_result


# 函数：调用 SiliconFlow API 来生成 LLM 响应
def call_llm(prompt, max_tokens=512, temperature=0.7, step_name="", expect_json=False):
    """
    调用 SiliconFlow API 来处理给定的提示。
    参数:
    prompt (str): 输入的提示文本。
    max_tokens (int): 最大生成的 token 数量，默认 512。
    temperature (float): 生成的温度参数，默认 0.7。
    step_name (str): 当前步骤名称，用于日志记录。
    expect_json (bool): 是否期望返回 JSON 格式，默认 False。
    返回:
    str: API 返回的响应内容。
    异常:
    如果 API 调用失败，抛出异常。
    """
    print(f"{'-' * 80}")
    print(f"🤖 LLM CALL - {step_name}")
    print(f"📤 PROMPT SENT:\n{prompt}")
    print(f"{'-' * 80}")

    headers = {
        "Authorization": f"Bearer {config.SF_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": config.SF_LLM_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(config.SF_API_LLM_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()["choices"][0]["message"]["content"].strip()
        print(f"📥 RESPONSE RECEIVED:\n{result}")
        print(f"{'=' * 80}")

        # 如果期望 JSON 格式，进行清理和解析
        if expect_json:
            return clean_and_parse_json_response(result, step_name)
        else:
            return result
    else:
        error_msg = f"API call failed: {response.text}"
        print(f"❌ ERROR: {error_msg}")
        print(f"{'=' * 80}")
        raise Exception(error_msg)


# 函数：初始化检索器
def initialize_retrievers():
    """
    初始化数据加载器和各种检索器。
    返回:
    tuple: 包含 BM25 检索器、BGE 检索器、Qwen3 检索器、BGE 重排序器 和 文档 ID 到文本的映射。
    """
    print("🔄 Initializing retrievers...")
    data_loader = HQSmallDataLoader(config.BASE_DATA_DIR)
    all_doc_ids, all_documents = data_loader.load_collection(config.COLLECTION_PATH)
    doc_id_to_text = dict(zip(all_doc_ids, all_documents))

    if config.SF_API_KEY:
        bge_reranker = BGEReranker(api_key=config.SF_API_KEY)
        bm25_retriever = BM25Retriever()
        qwen3_retriever = Qwen3Retriever(api_key=config.SF_API_KEY)
        bge_retriever = BGERetriever(api_key=config.SF_API_KEY)
    else:
        bge_reranker = BGEReranker()
        bm25_retriever = BM25Retriever()
        qwen3_retriever = Qwen3Retriever()
        bge_retriever = BGERetriever()

    bm25_retriever.load_index(config.BM25_INDEX_PATH)
    bge_retriever.load_index(config.BGE_INDEX_DIR)
    qwen3_retriever.load_index(config.QWEN_INDEX_DIR)
    print("✅ Retrievers initialized successfully")
    return bm25_retriever, bge_retriever, qwen3_retriever, bge_reranker, doc_id_to_text


# 函数：检索和格式化文档
def retrieve_documents(query, bm25_retriever, bge_retriever, bge_reranker, doc_id_to_text, retrieval_top_k=50,
                       rerank_top_k=10):
    """
    使用混合检索和重排序来检索文档。
    参数:
    query (str): 查询字符串。
    bm25_retriever: BM25 检索器实例。
    bge_retriever: BGE 检索器实例。
    bge_reranker: BGE 重排序器实例。
    doc_id_to_text (dict): 文档 ID 到文本的映射。
    retrieval_top_k (int): 初始检索的 top k 值，默认 50。
    rerank_top_k (int): 重排序后的 top k 值，默认 10。
    返回:
    tuple: 包含检索到的文档文本列表和文档 ID 列表。
    """
    print(f"🔍 RETRIEVING DOCUMENTS FOR QUERY: '{query}'")
    print(f"Retrieval top_k: {retrieval_top_k}, Rerank top_k: {rerank_top_k}")

    results = hybrid_retrieve_and_rerank(
        query=query,
        first_retriever=bm25_retriever,
        second_retriever=bge_retriever,
        reranker=bge_reranker,
        doc_id_to_text_map=doc_id_to_text,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k
    )
    doc_texts = [doc_id_to_text[doc_id] for doc_id, _ in results]
    doc_ids = [doc_id for doc_id, _ in results]

    print(f"✅ Retrieved {len(doc_texts)} documents")
    for i, (doc_id, text) in enumerate(zip(doc_ids, doc_texts)):
        print(f"   Document {i + 1} (ID: {doc_id}): {text[:100]}...")

    return doc_texts, doc_ids


# 管道函数：RAG 管道实现
def rag_pipeline(query):
    """
    RAG (Retrieval-Augmented Generation) 管道的主函数。
    处理查询，包括分解、检索、相关性检查、重写、生成答案、自检和合成。
    参数:
    query (str): 输入查询。
    返回:
    str: 最终生成的答案。
    """
    print(f"🎯 STARTING RAG PIPELINE FOR QUERY: '{query}'")

    bm25_retriever, bge_retriever, qwen3_retriever, bge_reranker, doc_id_to_text = initialize_retrievers()

    # 步骤 1: 查询分解
    print("=" * 100)
    print(f"📝 STEP 1: QUERY DECOMPOSITION")
    print(f"Original query: '{query}'")

    decomp_prompt = DECOMPOSITION_PROMPT.format(query=query)
    decomp_response = call_llm(decomp_prompt, step_name="QUERY DECOMPOSITION", expect_json=True)

    # 现在 decomp_response 已经是解析后的字典
    if isinstance(decomp_response, dict):
        needs_decomp = decomp_response.get("needs_decomposition", False)
        sub_queries = decomp_response.get("sub_queries", [])
        print(f"✅ Decomposition result: needs_decomposition={needs_decomp}, sub_queries={sub_queries}")
    else:
        print(f"❌ Unexpected response type for decomposition: {type(decomp_response)}")
        needs_decomp = False
        sub_queries = []

    queries = sub_queries if needs_decomp else [query]
    print(f"📋 Queries to process: {queries}")

    sub_answers = []

    for i, q in enumerate(queries):
        print("=" * 100)
        print(f"🔄 PROCESSING SUB-QUERY {i + 1}/{len(queries)}: '{q}'")

        current_query = q
        max_retries = 3
        is_relevant = False

        for attempt in range(max_retries):
            print(f"🔎 ATTEMPT {attempt + 1}/{max_retries}")
            print(f"   Current query: '{current_query}'")

            # 检索文档
            doc_texts, doc_ids = retrieve_documents(current_query, bm25_retriever, bge_retriever, bge_reranker,
                                                    doc_id_to_text)
            documents_str = "\n".join([f"Doc {i + 1}: {text}" for i, text in enumerate(doc_texts)])

            # 步骤 2: 相关性检查
            print("=" * 100)
            print(f"📊 STEP 2.{attempt + 1}: RELEVANCE CHECK")
            rel_prompt = RELEVANCE_CHECK_PROMPT.format(query=current_query, documents=documents_str)
            rel_response = call_llm(rel_prompt, step_name=f"RELEVANCE CHECK (Attempt {attempt + 1})", expect_json=True)

            # 现在 rel_response 已经是解析后的字典
            if isinstance(rel_response, dict):
                is_relevant = rel_response.get("is_relevant", False)
                reason = rel_response.get("reason", "")
                suggested_rewrite = rel_response.get("suggested_rewrite", "")
                print(f"   ✅ Relevance check result: is_relevant={is_relevant}, reason={reason}")
                if suggested_rewrite:
                    print(f"   💡 Suggested rewrite: {suggested_rewrite}")
            else:
                print(f"   ❌ Unexpected response type for relevance check: {type(rel_response)}")
                is_relevant = False
                reason = "Unexpected response type"
                suggested_rewrite = ""

            if is_relevant:
                print(f"   ✅ Documents are relevant, proceeding to answer generation")
                break
            else:
                print(f"   ⚠️ Documents not relevant, attempting query rewrite")
                # 重写查询
                rewrite_prompt = QUERY_REWRITE_PROMPT.format(original_query=current_query, reason=reason,
                                                             suggested_rewrite=suggested_rewrite)
                current_query = call_llm(rewrite_prompt, step_name=f"QUERY REWRITE (Attempt {attempt + 1})")
                print(f"   🔄 Rewrote query to: '{current_query}'")

        if not is_relevant:
            print(f"   ❌ Failed to find relevant documents after {max_retries} attempts")
            sub_answers.append("Insufficient information after retries.")
            continue

        # 步骤 3: 生成答案
        print("=" * 100)
        print(f"📝 STEP 3: GENERATE ANSWER")
        context = "\n\n".join(doc_texts)
        gen_prompt = GENERATE_ANSWER_PROMPT.format(query=current_query, context=context)
        gen_response = call_llm(gen_prompt, step_name="GENERATE ANSWER")

        if "\nEvidence: " in gen_response:
            answer, evidence = gen_response.split("\nEvidence: ", 1)
            print(f"✅ Answer generated with evidence")
            print(f"💡 Answer: {answer}")
            print(f"📚 Evidence: {evidence[:200]}...")
        else:
            answer = gen_response
            evidence = ""
            print(f"✅ Answer generated (no evidence separated)")
            print(f"💡 Answer: {answer}")

        # 自检
        print("=" * 100)
        print(f"✅ STEP 4: SELF-CHECK")
        self_check_prompt = SELF_CHECK_PROMPT.format(answer=answer, documents=documents_str)
        self_check_response = call_llm(self_check_prompt, step_name="SELF-CHECK", expect_json=True)

        # 现在 self_check_response 已经是解析后的字典
        if isinstance(self_check_response, dict):
            is_valid = self_check_response.get("is_valid", False)
            issues = self_check_response.get("issues", "")
            revised_answer = self_check_response.get("revised_answer", "")
            print(f"🔍 Self-check result: is_valid={is_valid}, issues={issues}")
            if revised_answer:
                print(f"📝 Revised answer: {revised_answer}")
        else:
            print(f"❌ Unexpected response type for self-check: {type(self_check_response)}")
            is_valid = False
            issues = "Unexpected response type"
            revised_answer = ""

        final_sub_answer = revised_answer if not is_valid else answer
        sub_answers.append(final_sub_answer)
        print(f"✅ Final sub-answer: {final_sub_answer}")

    # 如果分解了，则合成答案
    print("=" * 100)
    print(f"🎯 FINAL STEP: SYNTHESIZE ANSWERS")
    if needs_decomp and len(sub_answers) > 1:
        print(f"📦 Synthesizing {len(sub_answers)} sub-answers into final answer")
        sub_answers_str = "\n".join([f"Sub-answer {i + 1}: {answer}" for i, answer in enumerate(sub_answers)])
        synth_prompt = SYNTHESIZE_ANSWERS_PROMPT.format(original_query=query, sub_answers=sub_answers_str)
        final_answer = call_llm(synth_prompt, step_name="SYNTHESIZE ANSWERS")
        print(f"✅ Final synthesized answer ready")
    else:
        final_answer = sub_answers[0] if sub_answers else "No answer generated"
        print(f"✅ Using single answer as final answer")

    return final_answer


# 示例用法
if __name__ == "__main__":
    sample_query = "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?"
    print("🚀 STARTING RAG PIPELINE DEMO")
    print("=" * 100)
    answer = rag_pipeline(sample_query)
    print(f"🎉 FINAL RESULT")
    print("=" * 100)
    print(f"📝 Original Query: {sample_query}")
    print(f"💡 Final Answer: {answer}")
    print("=" * 100)