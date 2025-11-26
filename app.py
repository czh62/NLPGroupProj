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
# PipelineVisualizer（增强：动态生成子问题节点）
# ==========================================
class PipelineVisualizer:
    def __init__(self):
        self.base_nodes = [
            "Start", "Init Retrievers", "Query Decomposition"
        ]
        self.post_nodes = ["Synthesis", "End"]
        self.dynamic_nodes = []  # will hold per-subquery nodes
        self.graph = None
        self.current_step = None

    def build_graph_for_queries(self, queries):
        """根据子问题数量动态创建图节点"""
        self.dynamic_nodes = []
        # For each subquery, create a small linear chain: Sub i - Retrieval - Relevance - Generate - SelfCheck
        for i, q in enumerate(queries, start=1):
            label_prefix = f"Sub{i}"
            self.dynamic_nodes.append([
                f"{label_prefix}:Start",
                f"{label_prefix}:Retrieval",
                f"{label_prefix}:Relevance",
                f"{label_prefix}:Generate",
                f"{label_prefix}:SelfCheck",
                f"{label_prefix}:Done"
            ])

        # build a graphviz Digraph object and render initial state
        self._render_graph(active_node="Start")

    def _render_graph(self, active_node=None):
        g = graphviz.Digraph()
        g.attr(rankdir='TB')
        # base nodes
        for n in self.base_nodes:
            color = '#ff9f43' if n == active_node else 'white'
            fontcolor = 'white' if n == active_node else 'black'
            g.node(n, shape='box', style='rounded,filled', fillcolor=color, fontcolor=fontcolor)

        # connect base flow
        g.edge("Start", "Init Retrievers")
        g.edge("Init Retrievers", "Query Decomposition")

        # dynamic per-subquery nodes and connections
        for idx, node_group in enumerate(self.dynamic_nodes):
            # place subquery start node after Query Decomposition
            # connect chain
            prev = "Query Decomposition" if idx == 0 else self.dynamic_nodes[idx-1][-1]
            g.edge(prev, node_group[0])
            for a, b in zip(node_group, node_group[1:]):
                # highlight active
                color = '#ff9f43' if a == active_node or b == active_node else 'white'
                fontcolor = 'white' if a == active_node or b == active_node else 'black'
                g.node(a, shape='box', style='rounded,filled', fillcolor=('#ff9f43' if a==active_node else 'white'), fontcolor=('white' if a==active_node else 'black'))
                g.node(b, shape='box', style='rounded,filled', fillcolor=('#ff9f43' if b==active_node else 'white'), fontcolor=('white' if b==active_node else 'black'))
                g.edge(a, b)

        # connect last dynamic node to synthesis and end
        last_out = self.dynamic_nodes[-1][-1] if self.dynamic_nodes else "Query Decomposition"
        g.edge(last_out, "Synthesis")
        g.node("Synthesis", shape='box', style='rounded,filled', fillcolor=('#ff9f43' if active_node=="Synthesis" else 'white'), fontcolor=('white' if active_node=="Synthesis" else 'black'))
        g.node("End", shape='box', style='rounded,filled', fillcolor=('#ff9f43' if active_node=="End" else 'white'), fontcolor=('white' if active_node=="End" else 'black'))
        g.edge("Synthesis", "End")

        self.graph = g
        self.current_step = active_node

    def update(self, step_name, log_message=None):
        """外部调用：更新高亮节点并在页面上显示日志"""
        # step_name 需要与节点名一致（可为 Start, Init Retrievers, Query Decomposition,
        # Sub1:Retrieval, Sub1:Relevance, Sub1:Generate, Sub1:SelfCheck, Sub1:Done, Synthesis, End）
        try:
            self._render_graph(active_node=step_name)
        except Exception:
            # fallback: just render without active coloring
            self._render_graph(active_node=None)

        # 在 Streamlit 的 placeholder 中渲染
        if 'graph_placeholder' in st.session_state:
            with st.session_state['graph_placeholder'].container():
                st.graphviz_chart(self.graph, use_container_width=True)

        # 记录日志
        if log_message:
            if 'logs' not in st.session_state:
                st.session_state['logs'] = []
            st.session_state['logs'].append(f"**[{step_name}]**: {log_message}")
            if 'log_placeholder' in st.session_state:
                with st.session_state['log_placeholder'].container():
                    st.write(log_message)

# ==========================================
# JSON 清理 & LLM 调用 (保持你原有逻辑)
# ==========================================
def clean_and_parse_json_response(response_text, step_name="", visualizer=None):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    cleaned_text = response_text.strip()
    cleaned_text = re.sub(r'^```json\s*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'```\s*$', '', cleaned_text)
    cleaned_text = re.sub(r'^```\s*', '', cleaned_text)
    cleaned_text = re.sub(r'```\s*$', '', cleaned_text)
    cleaned_text = re.sub(r'^json\s*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = cleaned_text.strip()

    try:
        return json.loads(cleaned_text)
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
                return clean_and_parse_json_response(result, step_name, visualizer)
            else:
                return result
        else:
            raise Exception(f"API Error: {response.text}")
    except Exception as e:
        if visualizer:
            visualizer.update(step_name, f"❌ Error: {str(e)}")
        raise e

# ==========================================
# Cached retrievers 初始化
# ==========================================
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
        bge_reranker = BGEReranker()
        bm25_retriever = BM25Retriever()
        qwen3_retriever = Qwen3Retriever()
        bge_retriever = BGERetriever()

    bm25_retriever.load_index(config.BM25_INDEX_PATH)
    bge_retriever.load_index(config.BGE_INDEX_DIR)
    qwen3_retriever.load_index(config.QWEN_INDEX_DIR)

    return bm25_retriever, bge_retriever, qwen3_retriever, bge_reranker, doc_id_to_text

# ==========================================
# 检索函数：增强返回文档 id, text, score（如果可用）
# ==========================================
def retrieve_documents(query, bm25_retriever, bge_retriever, bge_reranker, doc_id_to_text,
                       retrieval_top_k=50, rerank_top_k=10, visualizer=None):
    if visualizer:
        visualizer.update("Retrieval", f"Retrieving for: '{query}'")

    results = hybrid_retrieve_and_rerank(
        query=query,
        first_retriever=bm25_retriever,
        second_retriever=bge_retriever,
        reranker=bge_reranker,
        doc_id_to_text_map=doc_id_to_text,
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k
    )
    # results is expected to be list of tuples (doc_id, score)
    doc_texts = []
    doc_ids = []
    doc_scores = []
    for doc_id, score in results:
        doc_ids.append(doc_id)
        doc_scores.append(score)
        doc_texts.append(doc_id_to_text.get(doc_id, "[NO TEXT FOUND]"))
    return doc_texts, doc_ids, doc_scores

# ==========================================
# 主 Pipeline（支持 UI 更新 & 动态流程图）
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
        sub_queries = decomp_response.get("sub_queries", []) or []
    else:
        needs_decomp = False
        sub_queries = []

    queries = sub_queries if needs_decomp and len(sub_queries) > 0 else [query]
    visualizer.build_graph_for_queries(queries)
    visualizer.update("Query Decomposition", f"Sub-queries: {queries}")

    sub_answers = []
    provenance = {}  # track which docs used per sub-answer: {sub_i: [doc_idx,...]}

    for i, q in enumerate(queries, start=1):
        node_prefix = f"Sub{i}"
        current_query = q
        max_retries = 3
        is_relevant = False

        with st.expander(f"Processing Sub-query {i}: {q}", expanded=True):
            for attempt in range(max_retries):
                step_retrieval_node = f"{node_prefix}:Retrieval"
                visualizer.update(step_retrieval_node, f"Attempt {attempt + 1}: retrieving for '{current_query}'")
                st.write(f"🔄 **Attempt {attempt + 1}** — Query: `{current_query}`")

                # 检索
                doc_texts, doc_ids, doc_scores = retrieve_documents(current_query, bm25_retriever, bge_retriever, bge_reranker,
                                                                    doc_id_to_text, visualizer=visualizer)
                st.info(f"Retrieved {len(doc_texts)} documents.")
                # 显示文献与编号（折叠显示全部）
                with st.expander(f"📚 Retrieved Documents ({len(doc_texts)}) — Click to expand"):
                    for j, (doc_id, score, text) in enumerate(zip(doc_ids, doc_scores, doc_texts), start=1):
                        # show short snippet and allow expanding for full text
                        snippet = text[:500].replace("\n", " ")
                        st.markdown(f"**[{j}] Document ID:** `{doc_id}`  — score: `{score}`")
                        st.markdown(f"> {snippet}...")
                        with st.expander(f"Show full document [{j}]"):
                            st.code(text)

                # 相关性检查
                rel_node = f"{node_prefix}:Relevance"
                visualizer.update(rel_node, f"Checking relevance for attempt {attempt + 1}")
                documents_str = "\n".join([f"Doc {idx + 1}: {text}" for idx, text in enumerate(doc_texts)])
                rel_prompt = RELEVANCE_CHECK_PROMPT.format(query=current_query, documents=documents_str)
                rel_response = call_llm(rel_prompt, step_name="Relevance Check", expect_json=True, visualizer=visualizer)

                if isinstance(rel_response, dict):
                    is_relevant = rel_response.get("is_relevant", False)
                    reason = rel_response.get("reason", "")
                    suggested_rewrite = rel_response.get("suggested_rewrite", "")
                else:
                    is_relevant = False
                    reason = "Parsing failed"
                    suggested_rewrite = ""

                if is_relevant:
                    st.success("✅ Documents are relevant for this sub-query.")
                    # record provenance: choose top-k doc indices used (we'll take first rerank_top_k)
                    used_doc_indices = list(range(min(len(doc_ids), 10)))
                    provenance[f"sub_{i}"] = used_doc_indices
                    # move on to answer generation
                    break
                else:
                    st.warning(f"⚠️ Not relevant. Reason: {reason}")
                    visualizer.update(f"{node_prefix}:Generate", "Skipping generate; will try rewrite")
                    # Query rewrite
                    visualizer.update(f"{node_prefix}:Relevance", "Will rewrite query...")
                    rewrite_prompt = QUERY_REWRITE_PROMPT.format(original_query=current_query, reason=reason,
                                                                 suggested_rewrite=suggested_rewrite)
                    current_query = call_llm(rewrite_prompt, step_name="Query Rewrite", visualizer=visualizer)
                    st.write(f"🔁 Rewritten Query: `{current_query}`")

            if not is_relevant:
                st.error(f"❌ Sub-query {i} failed to find relevant documents after {max_retries} attempts.")
                sub_answers.append("Insufficient information.")
                continue

            # 生成答案
            gen_node = f"{node_prefix}:Generate"
            visualizer.update(gen_node, "Generating answer based on context...")
            context = "\n\n".join(doc_texts)
            gen_prompt = GENERATE_ANSWER_PROMPT.format(query=current_query, context=context)
            gen_response = call_llm(gen_prompt, step_name="Generate Answer", visualizer=visualizer)

            # 拆分 evidence（如果有）
            if "\nEvidence: " in gen_response:
                answer, evidence = gen_response.split("\nEvidence: ", 1)
            else:
                answer = gen_response
                evidence = ""

            st.markdown("**Generated Answer (raw):**")
            st.write(answer)
            if evidence:
                with st.expander("🔎 Evidence from LLM (separated)"):
                    st.write(evidence)

            # 自检
            self_node = f"{node_prefix}:SelfCheck"
            visualizer.update(self_node, "Verifying answer (self-check)...")
            self_check_prompt = SELF_CHECK_PROMPT.format(answer=answer, documents=documents_str)
            self_check_response = call_llm(self_check_prompt, step_name="Self-Check", expect_json=True, visualizer=visualizer)

            if isinstance(self_check_response, dict):
                is_valid = self_check_response.get("is_valid", False)
                revised_answer = self_check_response.get("revised_answer", "")
                issues = self_check_response.get("issues", "")
            else:
                is_valid = True
                revised_answer = ""
                issues = "Self-check parsing failed"

            if not is_valid and revised_answer:
                final_sub_answer = revised_answer
                st.warning("🔧 Answer revised by self-check.")
                st.write(final_sub_answer)
            else:
                final_sub_answer = answer

            sub_answers.append(final_sub_answer)
            st.markdown(f"**Sub-Answer {i} (final):**")
            st.write(final_sub_answer)

            # mark sub as done in graph
            visualizer.update(f"{node_prefix}:Done", f"Sub-query {i} done.")

    # 合成
    visualizer.update("Synthesis", "Synthesizing final answer...")
    if len(sub_answers) > 1:
        sub_answers_str = "\n".join([f"Sub-answer {i + 1}: {a}" for i, a in enumerate(sub_answers)])
        synth_prompt = SYNTHESIZE_ANSWERS_PROMPT.format(original_query=query, sub_answers=sub_answers_str)
        final_answer = call_llm(synth_prompt, step_name="Synthesis", visualizer=visualizer)
    else:
        final_answer = sub_answers[0] if sub_answers else "No answer generated"

    # 展示最终结果与引用映射
    visualizer.update("End", "Process Completed.")
    return {
        "final_answer": final_answer,
        "sub_answers": sub_answers,
        "provenance": provenance
    }

# ==========================================
# Streamlit 页面布局
# ==========================================
st.set_page_config(page_title="RAG Workflow Visualizer (Enhanced)", layout="wide")

st.title("🤖 Interactive RAG Pipeline (Enhanced)")
st.markdown("动态流程图 + 每次检索文献列表 + 子问题详情 + 最终结果来源映射")

# 侧边栏配置
with st.sidebar:
    st.header("Settings")
    st.write("Current Model:", getattr(config, "SF_LLM_MODEL_NAME", "unknown"))
    if st.button("Clear History"):
        st.session_state['logs'] = []
        st.rerun()

# 初始化 Session State
if 'logs' not in st.session_state:
    st.session_state['logs'] = []

# 布局：左侧流程图，右侧交互与日志
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Live Workflow")
    # 占位符用于动态更新流程图
    if 'graph_placeholder' not in st.session_state:
        st.session_state['graph_placeholder'] = st.empty()

    viz = PipelineVisualizer()
    viz._render_graph(active_node="Start")
    with st.session_state['graph_placeholder'].container():
        st.graphviz_chart(viz.graph, use_container_width=True)

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
                result = rag_pipeline_web(user_query, viz)
                final_answer = result['final_answer']
                sub_answers = result['sub_answers']
                provenance = result['provenance']

                with result_container:
                    st.success("🎉 Final Answer Generated!")
                    st.markdown("### 🔎 Final Answer")
                    st.write(final_answer)

                    st.markdown("### 🧾 Sub-Answers")
                    for i, sub in enumerate(sub_answers, start=1):
                        st.markdown(f"**Sub {i}:**")
                        st.write(sub)

                    st.markdown("### 📚 Provenance (which retrieved docs were used per sub-answer)")
                    if provenance:
                        for k, indices in provenance.items():
                            st.write(f"- **{k}** used documents: {', '.join([str(x+1) for x in indices])}")
                    else:
                        st.write("No provenance recorded.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                import traceback
                st.code(traceback.format_exc())

# 在页面底部显示历史详细日志
with st.expander("📜 View Detailed Execution Logs"):
    for log in st.session_state['logs']:
        st.write(log)
