import json
import re

import requests

import config  # 假设 config.py 存在，包含 BASE_DATA_DIR, SF_API_KEY 等配置
from BGEReranker import BGEReranker  # BGE 重排序器模块
from BM25Retriever import BM25Retriever  # BM25 检索器模块
from HQSmallDataLoader import HQSmallDataLoader  # HotpotQA 小型数据集加载器
from denseInstructionRetriever import Qwen3Retriever  # Qwen3 密集检索器
from denseRetriever import BGERetriever  # BGE 密集检索器
from hybridRetrieveRerank import hybrid_retrieve_and_rerank  # 混合检索和重排序函数
from prompts import DECOMPOSITION_PROMPT, REWRITE_SUBQUERIES_PROMPT, RELEVANCE_AND_REWRITE_PROMPT, \
    GENERATE_ANSWER_PROMPT, SELF_CHECK_PROMPT, SYNTHESIZE_ANSWERS_PROMPT  # 导入新的提示模板


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
    # print(f"🧹 Cleaning JSON response for {step_name}...")
    # print(f"📥 Raw response: {response_text}")

    # 尝试直接解析
    try:
        result = json.loads(response_text)
        # print(f"   ✅ Direct JSON parsing successful")
        return result
    except json.JSONDecodeError:
        # print(f"   ⚠️ Direct parsing failed, attempting to extract JSON from markdown code blocks")
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
    # print(f"   🔧 Cleaned text: {cleaned_text}")

    # 尝试解析清理后的文本
    try:
        result = json.loads(cleaned_text)
        # print(f"   ✅ Cleaned JSON parsing successful")
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
        "improved_query": ""
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
    # print(f"{'-' * 80}")
    # print(f"🤖 LLM CALL - {step_name}")
    # print(f"📤 PROMPT SENT:\n{prompt}")
    # print(f"{'-' * 80}")

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
        # print(f"📥 RESPONSE RECEIVED:\n{result}")
        # print(f"{'=' * 80}")

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


# 函数：检索和格式化文档（只取50分以上的结果）
def retrieve_documents(query, bm25_retriever, bge_retriever, bge_reranker, doc_id_to_text, retrieval_top_k=50,
                       rerank_top_k=10, min_score=50):
    """
    使用混合检索和重排序来检索文档，只返回分数高于阈值的文档。
    参数:
    query (str): 查询字符串。
    bm25_retriever: BM25 检索器实例。
    bge_retriever: BGE 检索器实例。
    bge_reranker: BGE 重排序器实例。
    doc_id_to_text (dict): 文档 ID 到文本的映射。
    retrieval_top_k (int): 初始检索的 top k 值，默认 50。
    rerank_top_k (int): 重排序后的 top k 值，默认 10。
    min_score (int): 最小分数阈值，默认 50。
    返回:
    tuple: 包含检索到的文档文本列表和文档 ID 列表。
    """
    print(f"🔍 RETRIEVING DOCUMENTS FOR QUERY: '{query}'")
    print(f"Retrieval top_k: {retrieval_top_k}, Rerank top_k: {rerank_top_k}, Min score: {min_score}")

    results = hybrid_retrieve_and_rerank(
        query=query,
        first_retriever=bm25_retriever,
        second_retriever=bge_retriever,
        reranker=bge_reranker,
        doc_id_to_text_map=doc_id_to_text,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k
    )

    # 过滤分数高于阈值的文档
    filtered_results = [(doc_id, score) for doc_id, score in results if score >= min_score]

    if not filtered_results:
        print(f"⚠️ No documents found with score >= {min_score}, using top document regardless of score")
        filtered_results = [results[0]] if results else []

    doc_texts = [doc_id_to_text[doc_id] for doc_id, _ in filtered_results]
    doc_ids = [doc_id for doc_id, _ in filtered_results]

    print(f"✅ Retrieved {len(doc_texts)} documents (score >= {min_score})")
    for i, (doc_id, (_, score)) in enumerate(zip(doc_ids, filtered_results)):
        print(f"   Document {i + 1} (ID: {doc_id}, Score: {score:.2f}): {doc_texts[i][:100]}...")

    return doc_texts, doc_ids


# 函数：重写子查询（基于依赖关系和前序答案）
def rewrite_subqueries(original_query, sub_queries, previous_answers):
    """
    根据依赖关系和前序答案重写子查询。
    参数:
    original_query (str): 原始复杂问题
    sub_queries (list): 子问题列表
    previous_answers (dict): 前序问题的答案
    返回:
    list: 重写后的查询列表
    """
    print(f"🔄 REWRITING SUBQUERIES WITH DEPENDENCIES")
    print(f"Original query: {original_query}")
    print(f"Sub queries: {json.dumps(sub_queries, indent=2, ensure_ascii=False)}")
    print(f"Previous answers: {previous_answers}")

    rewrite_prompt = REWRITE_SUBQUERIES_PROMPT.format(
        original_query=original_query,
        sub_queries_json=json.dumps(sub_queries, ensure_ascii=False),
        previous_answers_json=json.dumps(previous_answers, ensure_ascii=False)
    )

    rewrite_response = call_llm(rewrite_prompt, step_name="REWRITE SUBQUERIES", expect_json=True)

    if isinstance(rewrite_response, dict) and "rewritten_queries" in rewrite_response:
        rewritten_queries = rewrite_response["rewritten_queries"]
        print(f"✅ Subqueries rewritten successfully")
        for rq in rewritten_queries:
            print(
                f"   {rq['original_id']}: '{rq['original_query']}' -> '{rq['rewritten_query']}'")
        return rewritten_queries
    else:
        print(f"❌ Failed to rewrite subqueries, using original queries")
        # 返回默认的重写结果
        return [
            {
                "original_id": sq["id"],
                "original_query": sq["query"],
                "rewritten_query": sq["query"],
            }
            for sq in sub_queries
        ]


# 函数：处理单个查询（支持上下文依赖）
def process_single_query(query, bm25_retriever, bge_retriever, bge_reranker, doc_id_to_text, previous_answers=None):
    """
    处理单个查询，包括检索、相关性判断、重写和答案生成。
    参数:
    query (str): 查询字符串。
    bm25_retriever: BM25 检索器实例。
    bge_retriever: BGE 检索器实例。
    bge_reranker: BGE 重排序器实例。
    doc_id_to_text (dict): 文档 ID 到文本的映射。
    previous_answers (dict): 前序问题的答案，用于上下文依赖。
    返回:
    tuple: 包含生成的答案和最终使用的查询。
    """
    original_query = query  # 保存原始查询
    current_query = query
    final_query = query  # 保存最终使用的查询
    max_retries = 3

    # 构建上下文依赖信息
    context_dependency = ""
    if previous_answers:
        context_dependency = "前序问题答案：\n"
        for qid, ans in previous_answers.items():
            context_dependency += f"- {qid}: {ans}\n"

    for attempt in range(max_retries):
        print(f"{'-' * 60}")
        print(f"🔎 ATTEMPT {attempt + 1}/{max_retries}")
        print(f"   Current query: '{current_query}'")
        if previous_answers:
            print(f"   Context from previous answers: {previous_answers}")

        # 检索文档（只取50分以上的结果）
        doc_texts, doc_ids = retrieve_documents(current_query, bm25_retriever, bge_retriever, bge_reranker,
                                                doc_id_to_text)
        documents_str = "\n".join([f"Doc {i + 1}: {text}" for i, text in enumerate(doc_texts)])

        # 步骤 2: 相关性判断 + 查询重写（支持上下文依赖）
        print(f"📝 STEP 2: RELEVANCE AND REWRITE")
        rel_rewrite_prompt = RELEVANCE_AND_REWRITE_PROMPT.format(
            query=current_query,
            context_dependency=context_dependency,
            documents=documents_str
        )
        rel_rewrite_response = call_llm(
            rel_rewrite_prompt,
            step_name=f"RELEVANCE AND REWRITE (Attempt {attempt + 1})",
            expect_json=True
        )

        # 处理响应
        if isinstance(rel_rewrite_response, dict):
            is_relevant = rel_rewrite_response.get("is_relevant", False)
            reason = rel_rewrite_response.get("reason", "")
            improved_query = rel_rewrite_response.get("improved_query", "")
            needs_context = rel_rewrite_response.get("needs_context", False)
            print(
                f"   ✅ Relevance and rewrite result: is_relevant={is_relevant}, reason={reason}, needs_context={needs_context}")
            if improved_query:
                print(f"   💡 Improved query: {improved_query}")
        else:
            print(f"   ❌ Unexpected response type for relevance and rewrite: {type(rel_rewrite_response)}")
            is_relevant = False
            reason = "Unexpected response type"
            improved_query = ""
            needs_context = False

        # 如果需要上下文但当前查询没有包含，则整合前序答案
        if needs_context and previous_answers and not any(
                str(qid).lower() in current_query.lower() for qid in previous_answers.keys()):
            context_info = " ".join([f"{ans}" for ans in previous_answers.values()])
            current_query = f"{current_query} {context_info}"
            print(f"   🔄 Enhanced query with context: '{current_query}'")
            continue  # 重新尝试当前查询

        if is_relevant:
            print(f"   ✅ Documents are relevant, proceeding to answer generation")
            final_query = current_query  # 使用当前查询作为最终查询
            break
        else:
            print(f"   ⚠️ Documents not relevant, using improved query for next attempt")
            if improved_query and improved_query.strip():
                current_query = improved_query.strip()
                print(f"   🔄 Using improved query: '{current_query}'")
            else:
                print(f"   ⚠️ No improved query provided, using original query")
                current_query = query

    # 如果三次尝试后仍然不相关，使用原始查询直接检索并生成答案
    if not is_relevant:
        print(f"   ❌ Failed to find relevant documents after {max_retries} attempts")
        print(f"   🔄 FALLBACK: Using original query '{original_query}' for direct retrieval and answer generation")

        # 使用原始查询直接检索文档
        doc_texts, doc_ids = retrieve_documents(original_query, bm25_retriever, bge_retriever, bge_reranker,
                                                doc_id_to_text)
        documents_str = "\n".join([f"Doc {i + 1}: {text}" for i, text in enumerate(doc_texts)])

        # 构建前序答案上下文字符串
        previous_answers_str = ""
        if previous_answers:
            previous_answers_str = "相关背景信息：\n"
            for qid, ans in previous_answers.items():
                previous_answers_str += f"- {ans}\n"

        # 直接生成答案，不进行相关性检查
        context = "\n\n".join(doc_texts)
        gen_prompt = GENERATE_ANSWER_PROMPT.format(
            query=original_query,
            previous_answers=previous_answers_str,
            context=context
        )
        answer = call_llm(gen_prompt, step_name="FALLBACK ANSWER GENERATION")
        print(f"✅ Fallback answer generated: {answer}")

        # 跳过自检步骤，直接返回答案
        return answer, original_query

    # 步骤 3: 生成答案 - 支持上下文传递
    print(f"📝 STEP 3: GENERATE ANSWER")
    print(f"Using final query for answer generation: '{final_query}'")
    context = "\n\n".join(doc_texts)

    # 构建前序答案上下文字符串
    previous_answers_str = ""
    if previous_answers:
        previous_answers_str = "相关背景信息：\n"
        for qid, ans in previous_answers.items():
            previous_answers_str += f"- {ans}\n"

    gen_prompt = GENERATE_ANSWER_PROMPT.format(
        query=final_query,
        previous_answers=previous_answers_str,
        context=context
    )
    answer = call_llm(gen_prompt, step_name="GENERATE ANSWER")
    print(f"✅ Answer generated: {answer}")

    # 步骤 4: 答案自检 - 使用最终查询
    print(f"📝 STEP 4: SELF-CHECK")
    self_check_prompt = SELF_CHECK_PROMPT.format(
        query=final_query,  # 使用最终查询
        answer=answer,
        documents=documents_str
    )
    self_check_response = call_llm(self_check_prompt, step_name="SELF-CHECK", expect_json=True)

    # 处理自检响应
    if isinstance(self_check_response, dict):
        is_valid = self_check_response.get("is_valid", False)
        issues = self_check_response.get("issues", "")
        revised_answer = self_check_response.get("revised_answer", "")
        print(f"🔍 Self-check result: is_valid={is_valid}, issues={issues}")
        if revised_answer and revised_answer.strip():
            print(f"📝 Using revised answer: {revised_answer}")
            return revised_answer.strip(), final_query
    else:
        print(f"❌ Unexpected response type for self-check: {type(self_check_response)}")

    return answer, final_query


# 函数：按依赖关系排序子问题
def sort_subqueries_by_dependency(sub_queries):
    """
    根据依赖关系对子问题进行拓扑排序。
    参数:
    sub_queries (list): 子问题列表，每个元素包含 id, query, depends_on
    返回:
    list: 按依赖关系排序后的子问题列表
    """
    # 构建依赖图
    graph = {}
    id_to_query = {}

    for sq in sub_queries:
        qid = sq.get('id', '')
        graph[qid] = sq.get('depends_on', [])
        id_to_query[qid] = sq

    # 拓扑排序
    visited = set()
    temp_visited = set()
    result = []

    def visit(node):
        if node in temp_visited:
            raise ValueError(f"Circular dependency detected involving {node}")
        if node not in visited:
            temp_visited.add(node)
            for dep in graph.get(node, []):
                visit(dep)
            temp_visited.remove(node)
            visited.add(node)
            result.append(id_to_query[node])

    for node in graph:
        if node not in visited:
            visit(node)

    return result


# 管道函数：RAG 管道实现（支持依赖关系和子查询重写）
def rag_pipeline(query):
    """
    RAG (Retrieval-Augmented Generation) 管道的主函数。
    处理查询，包括分解、检索、相关性判断、重写、生成答案、自检和合成。
    支持依赖关系的子问题处理和子查询重写。
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

    # 处理分解响应
    if isinstance(decomp_response, dict):
        needs_decomp = decomp_response.get("needs_decomposition", False)
        sub_queries_raw = decomp_response.get("sub_queries", [])
        print(f"✅ Decomposition result: needs_decomposition={needs_decomp}")

        # 处理子问题结构
        processed_sub_queries = []
        if needs_decomp and sub_queries_raw:
            for i, sq in enumerate(sub_queries_raw):
                if isinstance(sq, dict):
                    # 新格式：包含 id, query, depends_on
                    processed_sub_queries.append(sq)
                else:
                    # 旧格式：只有 query 字符串
                    processed_sub_queries.append({
                        "query": sq,
                        "id": f"q{i + 1}",
                        "depends_on": []
                    })

            # 按依赖关系排序
            try:
                sorted_sub_queries = sort_subqueries_by_dependency(processed_sub_queries)
                print(f"📋 Sorted sub-queries by dependency:")
                for i, sq in enumerate(sorted_sub_queries):
                    deps = sq.get('depends_on', [])
                    print(f"   {i + 1}. [{sq['id']}] {sq['query']} (depends on: {deps})")
            except ValueError as e:
                print(f"⚠️ Dependency sorting failed: {e}, using original order")
                sorted_sub_queries = processed_sub_queries
        else:
            sorted_sub_queries = []
    else:
        print(f"❌ Unexpected response type for decomposition: {type(decomp_response)}")
        needs_decomp = False
        sorted_sub_queries = []

    # 决定处理哪些查询
    if needs_decomp and sorted_sub_queries:
        queries_to_process = sorted_sub_queries
    else:
        queries_to_process = [{"query": query, "id": "q1", "depends_on": []}]

    print(f"📋 Queries to process: {len(queries_to_process)}")

    # 按顺序处理子问题，支持依赖关系和子查询重写
    sub_answers = {}  # 存储所有子问题的答案，用于依赖传递
    sub_answers_with_dependencies = []  # 用于最终合成的结构化信息

    for i, sub_query in enumerate(queries_to_process):
        print("=" * 100)
        print(f"🔄 PROCESSING SUB-QUERY {i + 1}/{len(queries_to_process)}")
        print(f"   Query ID: {sub_query['id']}")
        print(f"   Query: '{sub_query['query']}'")
        print(f"   Depends on: {sub_query.get('depends_on', [])}")

        # 收集依赖问题的答案
        previous_answers = {}
        for dep_id in sub_query.get('depends_on', []):
            if dep_id in sub_answers:
                previous_answers[dep_id] = sub_answers[dep_id]
            else:
                print(f"   ⚠️ Warning: Dependency {dep_id} not found in answers")

        # 步骤 1.5: 子查询重写（如果需要）
        current_query_to_process = sub_query['query']

        if previous_answers and sub_query.get('depends_on', []):
            # 有依赖关系，需要重写查询
            rewritten_queries = rewrite_subqueries(query, [sub_query], previous_answers)
            if rewritten_queries:
                rewritten_query = rewritten_queries[0]  # 只有一个查询
                current_query_to_process = rewritten_query['rewritten_query']
                print(f"   🔄 Rewritten query: '{current_query_to_process}'")

        # 处理当前子问题（如果需要检索）
        answer, final_used_query = process_single_query(
            current_query_to_process,
            bm25_retriever,
            bge_retriever,
            bge_reranker,
            doc_id_to_text,
            previous_answers=previous_answers
        )

        # 存储答案
        sub_answers[sub_query['id']] = {
            "question": final_used_query,  # 最终使用的查询
            "answer": answer
        }

        # 记录用于合成的信息
        sub_answers_with_dependencies.append({
            "id": sub_query['id'],
            "query": final_used_query,
            "answer": answer,
            "depends_on": sub_query.get('depends_on', [])
        })

        print(f"✅ Sub-answer {sub_query['id']} completed: {answer[:100]}...")

    # 步骤 5: 多子答案合成（如果需要）
    print("=" * 100)
    print(f"🎯 FINAL STEP: SYNTHESIZE ANSWERS")

    if needs_decomp and len(sub_answers_with_dependencies) > 1:
        print(f"📦 Synthesizing {len(sub_answers_with_dependencies)} sub-answers into final answer")

        # 构建用于合成的结构化信息
        sub_answers_str = "\n\n".join([
            f"Sub-query {item['id']} (depends on {item['depends_on']}): {item['query']}\nAnswer: {item['answer']}"
            for item in sub_answers_with_dependencies
        ])

        synth_prompt = SYNTHESIZE_ANSWERS_PROMPT.format(
            original_query=query,
            sub_answers_with_dependencies=sub_answers_str
        )
        final_answer = call_llm(synth_prompt, step_name="SYNTHESIZE ANSWERS")
        print(f"✅ Final synthesized answer ready")
    else:
        # 单个答案的情况
        if sub_answers_with_dependencies:
            final_answer = sub_answers_with_dependencies[0]['answer']
        else:
            final_answer = "根据提供的资料无法确定"
        print(f"✅ Using single answer as final answer")

    return final_answer


# 示例用法
if __name__ == "__main__":
    sample_query = "What direction does the river that Austrolebias bellotti are found in flow?"
    print("🚀 STARTING RAG PIPELINE DEMO")
    print("=" * 100)
    answer = rag_pipeline(sample_query)
    print(f"🎉 FINAL RESULT")
    print("=" * 100)
    print(f"📝 Original Query: {sample_query}")
    print(f"💡 Final Answer: {answer}")
    print("=" * 100)