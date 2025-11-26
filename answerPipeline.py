import json
import requests
import config
from BGEReranker import BGEReranker
from BM25Retriever import BM25Retriever
from HQSmallDataLoader import HQSmallDataLoader
from denseInstructionRetriever import Qwen3Retriever
from denseRetriever import BGERetriever
from hybridRetrieveRerank import hybrid_retrieve_and_rerank
from prompts import DECOMPOSITION_PROMPT, RELEVANCE_CHECK_PROMPT, QUERY_REWRITE_PROMPT, GENERATE_ANSWER_PROMPT, \
    SELF_CHECK_PROMPT, SYNTHESIZE_ANSWERS_PROMPT


# Function to call SiliconFlow API
def call_llm(prompt, max_tokens=512, temperature=0.7):
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
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"API call failed: {response.text}")

# Initialize retrievers (adapted from main_hybrid)
def initialize_retrievers():
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

# Function to retrieve and format documents
def retrieve_documents(query, bm25_retriever, bge_retriever, bge_reranker, doc_id_to_text, retrieval_top_k=50, rerank_top_k=10):
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

# Pipeline function
def rag_pipeline(query):
    bm25_retriever, bge_retriever, qwen3_retriever, bge_reranker, doc_id_to_text = initialize_retrievers()

    # Step 1: Decomposition
    decomp_prompt = DECOMPOSITION_PROMPT.format(query=query)
    decomp_response = call_llm(decomp_prompt)
    try:
        decomp_json = json.loads(decomp_response)
        needs_decomp = decomp_json["needs_decomposition"]
        sub_queries = decomp_json["sub_queries"]
    except:
        needs_decomp = False
        sub_queries = []

    queries = sub_queries if needs_decomp else [query]
    sub_answers = []

    for q in queries:
        current_query = q
        max_retries = 3
        for attempt in range(max_retries):
            # Retrieve
            doc_texts, doc_ids = retrieve_documents(current_query, bm25_retriever, bge_retriever, bge_reranker, doc_id_to_text)
            documents_str = "\n".join([f"Doc {i+1}: {text}" for i, text in enumerate(doc_texts)])

            # Step 2: Relevance check
            rel_prompt = RELEVANCE_CHECK_PROMPT.format(query=current_query, documents=documents_str)
            rel_response = call_llm(rel_prompt)
            try:
                rel_json = json.loads(rel_response)
                is_relevant = rel_json["is_relevant"]
                reason = rel_json["reason"]
                suggested_rewrite = rel_json["suggested_rewrite"]
            except:
                is_relevant = False
                reason = "Parsing error"
                suggested_rewrite = ""

            if is_relevant:
                break
            else:
                # Rewrite query
                rewrite_prompt = QUERY_REWRITE_PROMPT.format(original_query=current_query, reason=reason, suggested_rewrite=suggested_rewrite)
                current_query = call_llm(rewrite_prompt)
                print(f"Rewrote query to: {current_query}")

        if not is_relevant:
            sub_answers.append("Insufficient information after retries.")
            continue

        # Step 3: Generate answer
        context = "\n\n".join(doc_texts)
        gen_prompt = GENERATE_ANSWER_PROMPT.format(query=current_query, context=context)
        gen_response = call_llm(gen_prompt)
        answer, evidence = gen_response.split("\nEvidence: ", 1) if "\nEvidence: " in gen_response else (gen_response, "")

        # Self-check
        self_check_prompt = SELF_CHECK_PROMPT.format(answer=answer, documents=documents_str)
        self_check_response = call_llm(self_check_prompt)
        try:
            self_check_json = json.loads(self_check_response)
            is_valid = self_check_json["is_valid"]
            issues = self_check_json["issues"]
            revised_answer = self_check_json["revised_answer"]
        except:
            is_valid = False
            issues = "Parsing error"
            revised_answer = ""

        final_sub_answer = revised_answer if not is_valid else answer
        sub_answers.append(final_sub_answer)

    # Synthesize if decomposed
    if needs_decomp:
        sub_answers_str = "\n".join(sub_answers)
        synth_prompt = SYNTHESIZE_ANSWERS_PROMPT.format(original_query=query, sub_answers=sub_answers_str)
        final_answer = call_llm(synth_prompt)
    else:
        final_answer = sub_answers[0]

    return final_answer

# Example usage
if __name__ == "__main__":
    sample_query = "Which airport is located in Maine, Sacramento International Airport or Knox County Regional Airport?"
    answer = rag_pipeline(sample_query)
    print("Final Answer:", answer)