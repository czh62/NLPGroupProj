import json
import re

import requests
import streamlit as st

# Local Modules
import config
from BGEReranker import BGEReranker
from BM25Retriever import BM25Retriever
from HQSmallDataLoader import HQSmallDataLoader
from denseInstructionRetriever import Qwen3Retriever
from denseRetriever import BGERetriever
from hybridRetrieveRerank import hybrid_retrieve_and_rerank
from model2vecRetriever import Model2VecRetriever
from multivectorRetrieval import MultiVectorRetriever
from prompts import (
    DECOMPOSITION_PROMPT, REWRITE_SUBQUERIES_PROMPT,
    RELEVANCE_AND_REWRITE_PROMPT, GENERATE_ANSWER_PROMPT,
    SELF_CHECK_PROMPT, SYNTHESIZE_ANSWERS_PROMPT
)


# ==========================================
# Pure LLM Call (No intermediate display)
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
        raise Exception(f"API Error: {response.text}")

    text = response.json()["choices"][0]["message"]["content"].strip()
    if not expect_json:
        return text

    # JSON Cleaning and Parsing
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
# Resource Loader (Unified Retriever Management)
# ==========================================
@st.cache_resource
def get_resources():
    # 1. Load Document Data
    loader = HQSmallDataLoader(config.BASE_DATA_DIR)
    ids, docs = loader.load_collection(config.COLLECTION_PATH)
    doc_map = dict(zip(ids, docs))

    # 2. Initialize Reranker
    bge_reranker = BGEReranker(api_key=config.SF_API_KEY)

    # 3. Initialize Retrievers
    retrievers = {}

    # BM25
    try:
        bm25 = BM25Retriever()
        bm25.load_index(config.BM25_INDEX_PATH)
        retrievers["BM25"] = bm25
    except Exception as e:
        print(f"BM25 init failed: {e}")

    # BGE Dense
    try:
        bge = BGERetriever(api_key=config.SF_API_KEY)
        bge.load_index(config.BGE_INDEX_DIR)
        retrievers["BGE"] = bge
    except Exception as e:
        print(f"BGE init failed: {e}")

    # Model2Vec
    try:
        # Assuming model name/path is configured in config for Model2Vec
        m2v = Model2VecRetriever(model_name=config.MODEL2VEC_MODEL_NAME)
        m2v.load_index(config.MODEL2VEC_INDEX_PATH)
        retrievers["Model2Vec"] = m2v
    except Exception as e:
        print(f"Model2Vec init failed: {e}")

    # Qwen3
    try:
        qwen = Qwen3Retriever(api_key=config.SF_API_KEY)
        qwen.load_index(config.QWEN_INDEX_DIR)
        retrievers["Qwen3"] = qwen
    except Exception as e:
        print(f"Qwen3 init failed: {e}")

    # MultiVector
    try:
        multi = MultiVectorRetriever(chunk_size=100, chunk_overlap=20, api_key=config.SF_API_KEY)
        multi.load_index(config.MULTI_VECTOR_INDEX_DIR)
        retrievers["MultiVector"] = multi
    except Exception as e:
        print(f"MultiVector init failed: {e}")

    return retrievers, bge_reranker, doc_map


# ==========================================
# Main Pipeline
# ==========================================
def rag_pipeline(query, retriever_config):
    # ====================== Main Process Tracker ======================
    st.markdown(f"### Original Question: **{query}**")

    # Get Resources
    retrievers_map, bge_reranker, doc_map = get_resources()

    # Parse Current Configuration
    mode = retriever_config["mode"]
    selected_instances = []

    if mode == "single":
        r_name = retriever_config["selected"][0]
        selected_instances = [retrievers_map[r_name]]
        st.info(f"Current Mode: **Single Retrieval** | Using Retriever: `{r_name}`")
    else:
        r_name1 = retriever_config["selected"][0]
        r_name2 = retriever_config["selected"][1]
        selected_instances = [retrievers_map[r_name1], retrievers_map[r_name2]]
        st.info(f"Current Mode: **Hybrid Retrieval (Hybrid + Rerank)** | Combination: `{r_name1}` + `{r_name2}`")

    # Top-level navigation + Progress via st.tabs
    tab_decomp, tab_process, tab_final = st.tabs(["1. Query Decomposition", "2. Sub-query Processing", "3. Final Synthesis"])

    answers_dict = {}
    processed_subs = []

    # ====================== Tab 1: Query Decomposition ======================
    with tab_decomp:
        st.write("**Original Question**")
        st.info(query)

        with st.spinner("Decomposing query..."):
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
            st.success(f"Successfully decomposed into **{len(sub_queries)} sub-queries** (dependencies supported)")
        else:
            sub_queries = [{"id": "q1", "query": query, "depends_on": []}]
            st.info("No decomposition needed; processing as a single query.")

        # Display decomposition results nicely
        rows = []
        for sq in sub_queries:
            deps = " → ".join(sq.get("depends_on", [])) if sq.get("depends_on") else "None"
            rows.append({
                "Sub-query ID": sq['id'],
                "Content": sq['query'],
                "Dependencies": deps
            })
        st.table(rows)

    # ====================== Tab 2: Sub-query Processing ======================
    with tab_process:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, sq in enumerate(sub_queries):
            sq_id = sq["id"]
            progress_bar.progress((idx + 1) / len(sub_queries))
            status_text.text(f"Processing: {sq_id} - {sq['query'][:60]}...")

            with st.expander(f"Sub-query {sq_id}: {sq['query']}", expanded=(idx < 3)):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Original Sub-query**")
                    st.code(sq['query'], language=None)
                with col2:
                    deps = sq.get("depends_on", [])
                    if deps:
                        prev = {d: answers_dict[d]["answer"] for d in deps if d in answers_dict}
                        st.markdown("**Injected Dependency Answers**")
                        for k, v in prev.items():
                            st.caption(f"**{k}**: {v[:100]}...")
                    else:
                        st.success("No dependencies")

                # === Sub-query Rewriting ===
                current_q = sq["query"]
                if sq.get("depends_on"):
                    with st.spinner("Rewriting query based on dependencies..."):
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
                            st.markdown("**Rewritten Query**")
                            st.success(current_q)

                final_q = current_q

                # === Iterative Retrieval ===
                best_docs = None
                best_ids = None
                best_scores = None
                relevant = False

                for attempt in range(1, 4):
                    st.markdown(f"#### Retrieval Attempt #{attempt} (Query: `{final_q}`)")

                    with st.spinner(f"Retrieving..."):
                        results = []

                        # >>>>>> Core Retrieval Dispatch Logic <<<<<<
                        if mode == "hybrid":
                            # Hybrid Retrieval: Must use Reranker
                            results = hybrid_retrieve_and_rerank(
                                query=final_q,
                                first_retriever=selected_instances[0],
                                second_retriever=selected_instances[1],
                                reranker=bge_reranker,
                                doc_id_to_text_map=doc_map,
                                retrieval_top_k=50,
                                rerank_top_k=15
                            )
                            # In hybrid mode, scores are usually normalized, use threshold filtering
                            filtered = [(did, score) for did, score in results if score >= 50]
                            if not filtered and results:
                                filtered = results[:5]  # Fallback
                        else:
                            # Single Retrieval: Direct call to .query()
                            retriever = selected_instances[0]
                            # Get Top 15
                            raw_results = retriever.query(final_q, top_k=15)
                            results = raw_results
                            # Single retrieval scores are not normalized, threshold not applicable, take all Top K
                            filtered = results

                        # Unpack results
                        if filtered:
                            best_ids, best_scores = zip(*filtered)
                            best_docs = [doc_map[did] for did in best_ids]
                        else:
                            best_ids, best_scores, best_docs = [], [], []

                    # Highlight Retrieval Results
                    if best_scores:
                        display_count = min(5, len(best_scores))
                        score_cols = st.columns(display_count)
                        for i in range(display_count):
                            score = best_scores[i]
                            with score_cols[i]:
                                # Only Hybrid (with Reranker) is suitable for color grading
                                if mode == "hybrid":
                                    if score >= 70:
                                        st.metric(f"Doc {i + 1}", f"{score:.1f}", delta="High Relevance")
                                    elif score >= 60:
                                        st.metric(f"Doc {i + 1}", f"{score:.1f}", delta="Medium")
                                    else:
                                        st.metric(f"Doc {i + 1}", f"{score:.1f}", delta="Low Relevance")
                                else:
                                    # Single retrieval shows raw scores
                                    st.metric(f"Doc {i + 1}", f"{score:.4f}", delta="Raw Score")

                    with st.expander(f"View Top {len(best_docs)} Document Contents", expanded=False):
                        for i, (doc_id, text, score) in enumerate(zip(best_ids, best_docs, best_scores)):
                            st.markdown(f"**Doc {i + 1}** | ID: `{doc_id}` | **Score: {score:.4f}**")
                            st.caption(text[:800] + ("..." if len(text) > 800 else ""))
                            st.divider()

                    # Relevance Check (LLM Check)
                    if not best_docs:
                        st.warning("No documents retrieved.")
                        relevant = False
                    else:
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
                        reason = rel_resp.get("reason", "None")

                    if relevant:
                        st.success(f"Attempt #{attempt} successful! Documents are relevant.")
                        break
                    else:
                        st.warning(f"Attempt #{attempt} irrelevant: {reason}")
                        new_q = rel_resp.get("improved_query", "").strip() if 'rel_resp' in locals() else ""
                        if new_q and attempt < 3:
                            final_q = new_q
                            st.info(f"→ Continuing with improved query: {new_q}")
                        else:
                            st.info("No further improvements; max attempts reached.")

                # Fallback
                if not best_docs and results:
                    st.error("Multiple attempts failed to meet criteria; forcing use of last results.")
                    top5 = results[:5]
                    best_ids = [x[0] for x in top5]
                    best_docs = [doc_map[did] for did in best_ids]

                # === Answer Generation + Self-Check ===
                with st.spinner("Generating Answer + Self-Checking..."):
                    context = "\n\n".join(best_docs) if best_docs else "Unable to retrieve relevant information."
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
                        st.warning("Self-check found issues → Automatically revising")
                        answer = check.get("revised_answer", answer)

                st.markdown("**Final Answer**")
                st.success(answer)

                answers_dict[sq_id] = {"answer": answer, "final_query": final_q}
                processed_subs.append({"id": sq_id, "query": final_q, "answer": answer})

        progress_bar.empty()
        status_text.empty()

    # ====================== Tab 3: Final Synthesis ======================
    with tab_final:
        st.markdown("### Final Answer")
        if len(processed_subs) > 1:
            with st.spinner("Synthesizing multiple sub-answers..."):
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
            st.success("Multi-answer synthesis complete.")
        else:
            final_answer = processed_subs[0]["answer"] if processed_subs else "Unable to generate answer."

        st.markdown(f"<h2 style='text-align: center; color: #1E90FF;'>{final_answer}</h2>", unsafe_allow_html=True)
        st.text_area("Copy Final Answer", final_answer, height=200)

        return final_answer


# ==========================================
# Streamlit Interface
# ==========================================
st.set_page_config(page_title="NLP GROUP WORK", layout="centered")
st.title("NLP GROUP WORK")
st.markdown("""
**Features**: Multi-hop Decomposition → Sub-query Rewriting → Flexible Retrieval Config → Iterative Optimization → Self-Check
""")

# Get loaded retrievers list
retrievers_map, _, _ = get_resources()
available_retrievers = list(retrievers_map.keys())

with st.sidebar:
    st.header("System Info")
    st.write(f"LLM: `{config.SF_LLM_MODEL_NAME}`")
    st.write(f"Knowledge Base: `{config.COLLECTION_PATH}`")

    st.divider()
    st.header("Retrieval Strategy Configuration")

    # Retrieval Mode Selection
    retrieve_mode = st.radio(
        "Select Retrieval Mode",
        ("Single Retrieval", "Hybrid Retrieval (Hybrid + Rerank)"),
        index=0
    )

    selected_retrievers = []

    if retrieve_mode == "Single Retrieval":
        # Single selection dropdown
        choice = st.selectbox("Select Retriever", available_retrievers, index=0)
        selected_retrievers = [choice]
        st.caption("In single mode, raw scores from the retriever are used without re-ranking.")

    else:
        # Hybrid Retrieval Configuration
        st.markdown("**Select two different retrievers for hybrid mode:**")
        col1, col2 = st.columns(2)
        with col1:
            r1 = st.selectbox("Retriever A", available_retrievers, index=0, key="r1")
        with col2:
            # Default to second option if available
            default_idx = 1 if len(available_retrievers) > 1 else 0
            r2 = st.selectbox("Retriever B", available_retrievers, index=default_idx, key="r2")

        selected_retrievers = [r1, r2]
        st.caption("Hybrid mode merges results from both and uses **BGE Reranker** for re-ranking.")

    st.divider()
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared. Please refresh the page.")

query = st.text_input(
    "Enter your question",
    placeholder="E.g.: The second place finisher of the 2011 Gran Premio Santander d'Italia drove for who when he won the 2009 FIA Formula One World Championship?",
    value="The second place finisher of the 2011 Gran Premio Santander d'Italia drove for who when he won the 2009 FIA Formula One World Championship?"
)

if st.button("Start Reasoning", type="primary", use_container_width=True):
    if query.strip():
        # Build config object to pass to pipeline
        config_obj = {
            "mode": "single" if retrieve_mode == "Single Retrieval" else "hybrid",
            "selected": selected_retrievers
        }
        rag_pipeline(query, config_obj)
    else:
        st.warning("Please enter a question.")