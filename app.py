import streamlit as st
import json
import requests
import re
import time
import graphviz

# 假设这些模块都在本地目录
import config
from BGEReranker import BGEReranker
from BM25Retriever import BM25Retriever
from HQSmallDataLoader import HQSmallDataLoader
from denseInstructionRetriever import Qwen3Retriever
from denseRetriever import BGERetriever
from hybridRetrieveRerank import hybrid_retrieve_and_rerank
from prompts import DECOMPOSITION_PROMPT, RELEVANCE_CHECK_PROMPT, QUERY_REWRITE_PROMPT, GENERATE_ANSWER_PROMPT, \
    SELF_CHECK_PROMPT, SYNTHESIZE_ANSWERS_PROMPT


# ==========================================
# UI & 流程图 辅助类
# ==========================================

class PipelineVisualizer:
    def __init__(self):
        self.graph = graphviz.Digraph()
        self.graph.attr(rankdir='TB', size='10')
        self.current_step = "Start"
        self.logs = []

        # 定义流程图的结构（节点）
        self.nodes = [
            "Start", "Init Retrievers", "Query Decomposition",
            "Retrieval & Rerank", "Relevance Check", "Query Rewrite",
            "Generate Answer", "Self-Check", "Synthesis", "End"
        ]

        # 初始化所有节点为默认颜色
        for node in self.nodes:
            self.graph.node(node, shape='box', style='rounded,filled', fillcolor='white', color='black')

        # 定义边（连接关系）
        self.graph.edge("Start", "Init Retrievers")
        self.graph.edge("Init Retrievers", "Query Decomposition")
        self.graph.edge("Query Decomposition", "Retrieval & Rerank")
        self.graph.edge("Retrieval & Rerank", "Relevance Check")
        self.graph.edge("Relevance Check", "Generate Answer", label="Yes")
        self.graph.edge("Relevance Check", "Query Rewrite", label="No")
        self.graph.edge("Query Rewrite", "Retrieval & Rerank")
        self.graph.edge("Generate Answer", "Self-Check")
        self.graph.edge("Self-Check", "Synthesis")
        self.graph.edge("Synthesis", "End")

    def update(self, step_name, log_message=None):
        """更新当前活跃节点并显示日志"""
        self.current_step = step_name

        # 重新构建图以更新颜色
        new_graph = graphviz.Digraph()
        new_graph.attr(rankdir='TB')

        for node in self.nodes:
            if node == step_name:
                # 当前步骤高亮为橙色
                new_graph.node(node, shape='box', style='rounded,filled', fillcolor='#ff9f43', color='black',
                               fontcolor='white')
            else:
                new_graph.node(node, shape='box', style='rounded,filled', fillcolor='white', color='black')

        # 重新添加边
        new_graph.edge("Start", "Init Retrievers")
        new_graph.edge("Init Retrievers", "Query Decomposition")
        new_graph.edge("Query Decomposition", "Retrieval & Rerank")
        new_graph.edge("Retrieval & Rerank", "Relevance Check")
        new_graph.edge("Relevance Check", "Generate Answer", label="Relevant")
        new_graph.edge("Relevance Check", "Query Rewrite", label="Not Relevant")
        new_graph.edge("Query Rewrite", "Retrieval & Rerank")
        new_graph.edge("Generate Answer", "Self-Check")
        new_graph.edge("Self-Check", "Synthesis")
        new_graph.edge("Synthesis", "End")

        # 在Streamlit中渲染图表
        with st.session_state['graph_placeholder'].container():
            st.graphviz_chart(new_graph, use_container_width=True)

        # 记录并显示日志
        if log_message:
            st.session_state['logs'].append(f"**[{step_name}]**: {log_message}")
            with st.session_state['log_placeholder'].container():
                st.write(log_message)


# ==========================================
# 核心逻辑 (经过改造以适配UI更新)
# ==========================================

def clean_and_parse_json_response(response_text, step_name="", visualizer=None):
    # (保持原逻辑，简化print为pass，或使用visualizer记录)
    try:
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError:
        pass

    cleaned_text = response_text.strip()
    cleaned_text = re.sub(r'^```json\s*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'```\s*$', '', cleaned_text)
    cleaned_text = re.sub(r'^json\s*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = cleaned_text.strip()

    try:
        result = json.loads(cleaned_text)
        return result
    except json.JSONDecodeError:
        try:
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = cleaned_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass

    return {"is_relevant": False, "reason": "JSON parsing failed", "suggested_rewrite": ""}


def call_llm(prompt, max_tokens=512, temperature=0.7, step_name="", expect_json=False, visualizer=None):
    if visualizer:
        # 简单显示 Prompt 的前一部分，避免UI太乱
        visualizer.update(step_name, f"Sending Prompt to LLM...")

    headers = {
        "Authorization": f"Bearer {config.SF_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": config.SF_LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(config.SF_API_LLM_URL, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"].strip()
            if expect_json:
                return clean_and_parse_json_response(result, step_name)
            else:
                return result
        else:
            raise Exception(f"API Error: {response.text}")
    except Exception as e:
        if visualizer:
            visualizer.update(step_name, f"❌ Error: {str(e)}")
        raise e


# 使用 st.cache_resource 缓存检索器，避免每次点击按钮都重新加载模型
@st.cache_resource
def initialize_retrievers_cached():
    data_loader = HQSmallDataLoader(config.BASE_DATA_DIR)
    all_doc_ids, all_documents = data_loader.load_collection(config.COLLECTION_PATH)
    doc_id_to_text = dict(zip(all_doc_ids, all_documents))

    if config.SF_API_KEY:
        bge_reranker = BGEReranker(api_key=config.SF_API_KEY)
        bm25_retriever = BM25Retriever()
        qwen3_retriever = Qwen3Retriever(api_key=config.SF_API_KEY)
        bge_retriever = BGERetriever(api_key=config.SF_API_KEY)
    else:
        # Fallback handling
        bge_reranker = BGEReranker()
        bm25_retriever = BM25Retriever()
        qwen3_retriever = Qwen3Retriever()
        bge_retriever = BGERetriever()

    bm25_retriever.load_index(config.BM25_INDEX_PATH)
    bge_retriever.load_index(config.BGE_INDEX_DIR)
    qwen3_retriever.load_index(config.QWEN_INDEX_DIR)

    return bm25_retriever, bge_retriever, qwen3_retriever, bge_reranker, doc_id_to_text


def retrieve_documents(query, bm25_retriever, bge_retriever, bge_reranker, doc_id_to_text, retrieval_top_k=50,
                       rerank_top_k=10, visualizer=None):
    if visualizer:
        visualizer.update("Retrieval & Rerank", f"Retrieving for: '{query}'")

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
    return doc_texts, doc_ids


# ==========================================
# 主 Pipeline (修改版)
# ==========================================

def rag_pipeline_web(query, visualizer):
    visualizer.update("Start", "Starting RAG Pipeline...")

    # 初始化
    visualizer.update("Init Retrievers", "Loading retrievers (cached)...")
    bm25_retriever, bge_retriever, qwen3_retriever, bge_reranker, doc_id_to_text = initialize_retrievers_cached()

    # 1. 分解
    visualizer.update("Query Decomposition", f"Analyzing query: {query}")
    decomp_prompt = DECOMPOSITION_PROMPT.format(query=query)
    decomp_response = call_llm(decomp_prompt, step_name="Query Decomposition", expect_json=True, visualizer=visualizer)

    if isinstance(decomp_response, dict):
        needs_decomp = decomp_response.get("needs_decomposition", False)
        sub_queries = decomp_response.get("sub_queries", [])
    else:
        needs_decomp = False
        sub_queries = []

    queries = sub_queries if needs_decomp else [query]
    visualizer.update("Query Decomposition", f"Sub-queries: {queries}")

    sub_answers = []

    for i, q in enumerate(queries):
        current_query = q
        max_retries = 3
        is_relevant = False

        with st.expander(f"Processing Sub-query: {q}", expanded=True):
            for attempt in range(max_retries):
                st.write(f"🔄 **Attempt {attempt + 1}**")

                # 检索
                doc_texts, doc_ids = retrieve_documents(current_query, bm25_retriever, bge_retriever, bge_reranker,
                                                        doc_id_to_text, visualizer=visualizer)
                documents_str = "\n".join([f"Doc {i + 1}: {text}" for i, text in enumerate(doc_texts)])

                st.info(f"Retrieved {len(doc_texts)} documents.")

                # 2. 相关性检查
                visualizer.update("Relevance Check", f"Checking relevance for attempt {attempt + 1}")
                rel_prompt = RELEVANCE_CHECK_PROMPT.format(query=current_query, documents=documents_str)
                rel_response = call_llm(rel_prompt, step_name="Relevance Check", expect_json=True,
                                        visualizer=visualizer)

                if isinstance(rel_response, dict):
                    is_relevant = rel_response.get("is_relevant", False)
                    reason = rel_response.get("reason", "")
                    suggested_rewrite = rel_response.get("suggested_rewrite", "")
                else:
                    is_relevant = False
                    reason = "Parsing failed"
                    suggested_rewrite = ""

                if is_relevant:
                    st.success("✅ Documents are relevant.")
                    break
                else:
                    st.warning(f"⚠️ Not relevant. Reason: {reason}")
                    visualizer.update("Query Rewrite", "Rewriting query...")
                    rewrite_prompt = QUERY_REWRITE_PROMPT.format(original_query=current_query, reason=reason,
                                                                 suggested_rewrite=suggested_rewrite)
                    current_query = call_llm(rewrite_prompt, step_name="Query Rewrite", visualizer=visualizer)
                    st.write(f"New Query: {current_query}")

            if not is_relevant:
                sub_answers.append("Insufficient information.")
                continue

            # 3. 生成答案
            visualizer.update("Generate Answer", "Generating answer based on context...")
            context = "\n\n".join(doc_texts)
            gen_prompt = GENERATE_ANSWER_PROMPT.format(query=current_query, context=context)
            gen_response = call_llm(gen_prompt, step_name="Generate Answer", visualizer=visualizer)

            if "\nEvidence: " in gen_response:
                answer, evidence = gen_response.split("\nEvidence: ", 1)
            else:
                answer = gen_response

            # 4. 自检
            visualizer.update("Self-Check", "Verifying answer accuracy...")
            self_check_prompt = SELF_CHECK_PROMPT.format(answer=answer, documents=documents_str)
            self_check_response = call_llm(self_check_prompt, step_name="Self-Check", expect_json=True,
                                           visualizer=visualizer)

            if isinstance(self_check_response, dict):
                is_valid = self_check_response.get("is_valid", False)
                revised_answer = self_check_response.get("revised_answer", "")
            else:
                is_valid = True  # Default to trust if check fails
                revised_answer = ""

            final_sub_answer = revised_answer if (not is_valid and revised_answer) else answer
            sub_answers.append(final_sub_answer)
            st.markdown(f"**Sub-Answer:** {final_sub_answer}")

    # 5. 合成
    visualizer.update("Synthesis", "Synthesizing final answer...")
    if needs_decomp and len(sub_answers) > 1:
        sub_answers_str = "\n".join([f"Sub-answer {i + 1}: {answer}" for i, answer in enumerate(sub_answers)])
        synth_prompt = SYNTHESIZE_ANSWERS_PROMPT.format(original_query=query, sub_answers=sub_answers_str)
        final_answer = call_llm(synth_prompt, step_name="Synthesis", visualizer=visualizer)
    else:
        final_answer = sub_answers[0] if sub_answers else "No answer generated"

    visualizer.update("End", "Process Completed.")
    return final_answer


# ==========================================
# Streamlit 页面布局
# ==========================================

st.set_page_config(page_title="RAG Workflow Visualizer", layout="wide")

st.title("🤖 Interactive RAG Pipeline")
st.markdown("This tool visualizes the retrieval-augmented generation process step-by-step.")

# 侧边栏配置
with st.sidebar:
    st.header("Settings")
    st.write("Current Model:", config.SF_LLM_MODEL_NAME)
    if st.button("Clear History"):
        st.session_state['logs'] = []
        st.rerun()

# 初始化 Session State
if 'logs' not in st.session_state:
    st.session_state['logs'] = []

# 布局：左侧是流程图，右侧是交互和详细日志
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Live Workflow")
    # 占位符用于动态更新流程图
    if 'graph_placeholder' not in st.session_state:
        st.session_state['graph_placeholder'] = st.empty()

    # 初始化显示一个静态图
    viz = PipelineVisualizer()
    viz.update("Start")  # Render initial state

with col2:
    st.subheader("💬 Query & Process")

    user_query = st.text_input("Enter your question:",
                               "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?")

    start_btn = st.button("🚀 Start Search", type="primary")

    # 用于显示实时日志的占位符
    st.session_state['log_placeholder'] = st.empty()

    result_container = st.container()

    if start_btn and user_query:
        st.session_state['logs'] = []  # Clear old logs

        with st.spinner("Running RAG Pipeline..."):
            try:
                # 运行 Pipeline
                final_answer = rag_pipeline_web(user_query, viz)

                # 显示最终结果
                with result_container:
                    st.success("🎉 Final Answer Generated!")
                    st.markdown(f"### Answer:\n{final_answer}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                # 打印堆栈以便调试
                import traceback

                st.code(traceback.format_exc())

# 在页面底部显示历史详细日志
with st.expander("📜 View Detailed Execution Logs"):
    for log in st.session_state['logs']:
        st.write(log)