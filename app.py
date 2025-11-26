import json
import re

import requests
import streamlit as st

# 本地模块
import config
from BGEReranker import BGEReranker
from BM25Retriever import BM25Retriever
from HQSmallDataLoader import HQSmallDataLoader
from denseRetriever import BGERetriever
from hybridRetrieveRerank import hybrid_retrieve_and_rerank
from prompts import (
    DECOMPOSITION_PROMPT, REWRITE_SUBQUERIES_PROMPT,
    RELEVANCE_AND_REWRITE_PROMPT, GENERATE_ANSWER_PROMPT,
    SELF_CHECK_PROMPT, SYNTHESIZE_ANSWERS_PROMPT
)


# ==========================================
# 纯净的 LLM 调用（不显示任何中间状态）
# ==========================================
def call_llm(prompt, max_tokens=512, temperature=0.7, expect_json=False):
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
    response = requests.post(config.SF_API_LLM_URL, headers=headers, json=data, timeout=180)
    if response.status_code != 200:
        raise Exception(f"API 错误: {response.text}")

    text = response.json()["choices"][0]["message"]["content"].strip()
    if not expect_json:
        return text

    # JSON 清理解析
    cleaned = re.sub(r'^```json\s*|\s*```$', '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'^json\s*', '', cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except:
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except:
                pass
    return {"error": "JSON parsing failed"}


# ==========================================
# 缓存检索器
# ==========================================
@st.cache_resource
def get_retrievers():
    loader = HQSmallDataLoader(config.BASE_DATA_DIR)
    ids, docs = loader.load_collection(config.COLLECTION_PATH)
    doc_map = dict(zip(ids, docs))

    bge_reranker = BGEReranker(api_key=config.SF_API_KEY)
    bm25 = BM25Retriever()
    bge_ret = BGERetriever(api_key=config.SF_API_KEY)

    bm25.load_index(config.BM25_INDEX_PATH)
    bge_ret.load_index(config.BGE_INDEX_DIR)

    return bm25, bge_ret, bge_reranker, doc_map


# ==========================================
# 主流程（完全静默执行 + 结果精准展示）
# ==========================================
def rag_pipeline(query):
    # ====================== 主流程追踪栏 ======================
    st.markdown(f"### 原始问题：**{query}**")

    # 用 st.tabs 做顶级导航 + 进度条
    tab_decomp, tab_process, tab_final = st.tabs(["1. 查询分解", "2. 子问题逐个处理", "3. 最终答案合成"])

    bm25, bge_ret, bge_rerank, doc_map = get_retrievers()
    answers_dict = {}
    processed_subs = []

    # ====================== Tab 1: 查询分解 ======================
    with tab_decomp:
        st.write("**原始问题**")
        st.info(query)

        with st.spinner("正在进行查询分解..."):
            resp = call_llm(DECOMPOSITION_PROMPT.format(query=query), expect_json=True)

        needs_decomp = resp.get("needs_decomposition", False)
        raw_subs = resp.get("sub_queries", [])

        if needs_decomp and raw_subs:
            sub_queries = []
            for i, item in enumerate(raw_subs):
                if isinstance(item, dict):
                    sub_queries.append(item)
                else:
                    sub_queries.append({"id": f"q{i + 1}", "query": str(item), "depends_on": []})
            st.success(f"成功分解为 **{len(sub_queries)} 个子问题**（支持依赖关系）")
        else:
            sub_queries = [{"id": "q1", "query": query, "depends_on": []}]
            st.info("无需分解，直接作为单一问题处理")

        # 美观展示分解结果（关键表格）
        rows = []
        for sq in sub_queries:
            deps = " → ".join(sq.get("depends_on", [])) if sq.get("depends_on") else "无"
            rows.append({
                "子问题ID": sq['id'],
                "子问题内容": sq['query'],
                "依赖": deps
            })
        st.table(rows)

    # ====================== Tab 2: 子问题逐个处理 ======================
    with tab_process:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, sq in enumerate(sub_queries):
            sq_id = sq["id"]
            progress_bar.progress((idx + 1) / len(sub_queries))
            status_text.text(f"正在处理：{sq_id} - {sq['query'][:60]}...")

            # 每个子问题用独立的 expander（默认展开前3个）
            with st.expander(f"子问题 {sq_id}：{sq['query']}", expanded=(idx < 3)):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**原始子问题**")
                    st.code(sq['query'], language=None)
                with col2:
                    deps = sq.get("depends_on", [])
                    if deps:
                        prev = {d: answers_dict[d]["answer"] for d in deps if d in answers_dict}
                        st.markdown("**已注入依赖答案**")
                        for k, v in prev.items():
                            st.caption(f"**{k}**: {v[:100]}...")
                    else:
                        st.success("无依赖")

                # === 子查询重写 ===
                current_q = sq["query"]
                if sq.get("depends_on"):
                    with st.spinner("根据依赖重写查询..."):
                        rewrite_resp = call_llm(
                            REWRITE_SUBQUERIES_PROMPT.format(
                                original_query=query,
                                sub_queries_json=json.dumps([sq], ensure_ascii=False),
                                previous_answers_json=json.dumps(
                                    {d: answers_dict[d]["answer"] for d in sq.get("depends_on", []) if
                                     d in answers_dict},
                                    ensure_ascii=False
                                )
                            ),
                            expect_json=True
                        )
                        rewritten = rewrite_resp.get("rewritten_queries", [])
                        if rewritten:
                            current_q = rewritten[0]["rewritten_query"]
                            st.markdown("**重写后查询**")
                            st.success(current_q)

                final_q = current_q

                # === 迭代检索（最多3轮）===
                best_docs = None
                best_ids = None
                best_scores = None
                relevant = False

                for attempt in range(1, 4):
                    st.markdown(f"#### 第 {attempt} 次检索（查询：`{final_q}`）")

                    with st.spinner(f"混合检索 + 重排序中..."):
                        results = hybrid_retrieve_and_rerank(
                            query=final_q,
                            first_retriever=bm25,
                            second_retriever=bge_ret,
                            reranker=bge_rerank,
                            doc_id_to_text_map=doc_map,
                            retrieval_top_k=50,
                            rerank_top_k=15
                        )
                        filtered = [(did, score) for did, score in results if score >= 50]
                        if not filtered and results:
                            filtered = results[:5]  # 兜底取最高5篇

                        best_ids, best_scores = zip(*filtered)
                        best_docs = [doc_map[did] for did in best_ids]

                    # 高亮展示检索结果
                    score_cols = st.columns(len(filtered[:5]))
                    for i, (doc_id, score) in enumerate(zip(best_ids[:5], best_scores[:5])):
                        with score_cols[i]:
                            if score >= 70:
                                st.metric(f"Doc {i + 1}", f"{score:.1f}", delta="高相关")
                            elif score >= 60:
                                st.metric(f"Doc {i + 1}", f"{score:.1f}", delta="中等")
                            else:
                                st.metric(f"Doc {i + 1}", f"{score:.1f}", delta="低相关")

                    with st.expander(f"查看全部 {len(filtered)} 篇高分文档内容", expanded=False):
                        for i, (doc_id, text, score) in enumerate(zip(best_ids, best_docs, best_scores)):
                            st.markdown(f"**Doc {i + 1}** | ID: `{doc_id}` | **Rerank 分数: {score:.2f}**")
                            st.caption(text[:1000] + ("..." if len(text) > 1000 else ""))
                            st.divider()

                    # 相关性判断
                    ctx_preview = "\n\n".join([f"Doc {i + 1}: {d[:500]}..." for i, d in enumerate(best_docs[:5])])
                    rel_resp = call_llm(
                        RELEVANCE_AND_REWRITE_PROMPT.format(
                            query=final_q,
                            context_dependency="\n".join(
                                [answers_dict[d]["answer"] for d in sq.get("depends_on", []) if d in answers_dict]),
                            documents=ctx_preview
                        ),
                        expect_json=True
                    )
                    relevant = rel_resp.get("is_relevant", False)
                    reason = rel_resp.get("reason", "无")

                    if relevant:
                        st.success(f"第 {attempt} 次检索成功！文档足够相关")
                        break
                    else:
                        st.warning(f"第 {attempt} 次不相关：{reason}")
                        new_q = rel_resp.get("improved_query", "").strip()
                        if new_q and attempt < 3:
                            final_q = new_q
                            st.info(f"→ 采用改进查询继续：{new_q}")
                        else:
                            st.info("不再改进，已达最大轮次")

                # 兜底
                if not relevant:
                    st.error("多次检索仍未达标，使用最高分文档强制生成")
                    top10 = results[:10]
                    best_ids = [x[0] for x in top10]
                    best_scores = [x[1] for x in top10]
                    best_docs = [doc_map[did] for did in best_ids]

                # === 答案生成 + 自检 ===
                with st.spinner("生成答案 + 自检..."):
                    context = "\n\n".join(best_docs)
                    prev_ctx = "\n".join(
                        [f"{k}: {v}" for k, v in answers_dict.items() if k in sq.get("depends_on", [])])
                    answer = call_llm(GENERATE_ANSWER_PROMPT.format(
                        query=final_q,
                        previous_answers=prev_ctx,
                        context=context
                    ))

                    check = call_llm(SELF_CHECK_PROMPT.format(query=final_q, answer=answer, documents=context),
                                     expect_json=True)
                    if not check.get("is_valid", True):
                        st.warning("自检发现问题 → 自动修正")
                        answer = check.get("revised_answer", answer)

                st.markdown("**最终答案**")
                st.success(answer)

                answers_dict[sq_id] = {"answer": answer, "final_query": final_q}
                processed_subs.append({"id": sq_id, "query": final_q, "answer": answer})

        progress_bar.empty()
        status_text.empty()

    # ====================== Tab 3: 最终答案合成 ======================
    with tab_final:
        st.markdown("### 最终答案")
        if len(processed_subs) > 1:
            with st.spinner("多子答案合成中..."):
                synth_text = "\n\n".join([
                    f"【{s['id']}】 {s['query']}\n→ {s['answer']}"
                    for s in processed_subs
                ])
                final_answer = call_llm(
                    SYNTHESIZE_ANSWERS_PROMPT.format(
                        original_query=query,
                        sub_answers_with_dependencies=synth_text
                    )
                )
            st.success("多答案合成完成")
        else:
            final_answer = processed_subs[0]["answer"] if processed_subs else "无法生成答案"

        # 大字体高亮最终输出
        st.markdown(f"<h2 style='text-align: center; color: #1E90FF;'>{final_answer}</h2>", unsafe_allow_html=True)

        # 可复制文本框
        st.text_area("复制最终答案", final_answer, height=200)

        return final_answer


# ==========================================
# Streamlit 界面
# ==========================================
st.set_page_config(page_title="高级 RAG 系统（极简专业版）", layout="centered")
st.title("高级 RAG 系统")
st.markdown("""
**特性**：多跳依赖分解 → 子查询重写 → 迭代检索（完整文档展示）→ 自检 → 答案合成  
**界面特点**：无冗余提示、结果清晰、适合演示与调试
""")

with st.sidebar:
    st.header("系统信息")
    st.write(f"LLM: `{config.SF_LLM_MODEL_NAME}`")
    st.write(f"知识库: `{config.COLLECTION_PATH}`")
    if st.button("清空缓存"):
        st.cache_resource.clear()
        st.success("缓存已清空")

query = st.text_input(
    "请输入问题",
    placeholder="例如：Austrolebias bellotti 所在的河流是朝哪个方向流的？",
    value="What direction does the river that Austrolebias bellotti are found in flow?"
)

if st.button("开始推理", type="primary", use_container_width=True):
    if query.strip():
        rag_pipeline(query)
    else:
        st.warning("请输入问题")