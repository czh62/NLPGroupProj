DECOMPOSITION_PROMPT = """
You are a professional query decomposition expert. Your ONLY job is to judge whether decomposition genuinely helps retrieval and final answer quality.

Strict rules — violate any one and you fail:
- Return needs_decomposition = false for ANY question that is simple, factual, single-entity, single-condition, or can be answered accurately with one search.
- ONLY decompose when the question clearly contains ≥2 of the following traits:
  • Multiple distinct entities that need separate lookups
  • Explicit comparison between two or more things
  • Causal reasoning (“why”, “how did X lead to Y”)
  • Multi-hop logic requiring intermediate facts
  • Timeline or sequence of events
  • Combination of conditions that cannot be searched effectively in one query
- NEVER decompose just because the question is long or has multiple sentences.
- NEVER create sub-questions that are minor rephrasings or trivial variants.
- Maximum 6 sub-questions, prefer 2–4 when decomposition is truly needed.
- Each sub-question must be independently searchable and add unique value.

User question: {query}

Respond strictly in JSON only, no explanations whatsoever:
{{
    "needs_decomposition": true/false,
    "sub_queries": [
        {{
            "query": "Complete, self-contained sub-question",
            "id": "Q1",
            "depends_on": []
        }}
    ]
}}
""".strip()


REWRITE_SUBQUERIES_PROMPT = """
You are a query optimization expert. Based on dependencies between sub-questions and previously obtained preceding answers, rewrite dependent sub-questions into complete, standalone queries that can be searched independently.

Input:
- Original complex question: {original_query}
- All sub-questions list (including dependencies): {sub_queries_json}
- Previously answered preceding answers (keyed by sub-query id): {previous_answers_json}

Rewriting rules:
1. Sub-questions with empty depends_on remain unchanged
2. For sub-questions with dependencies, naturally incorporate the preceding answers into the query
3. Rewritten queries must be concise, search-friendly, and preserve original intent

Output strictly in the following JSON format (JSON only):
{{
    "rewritten_queries": [
        {{
            "original_id": "q1",
            "original_query": "original sub-question text",
            "rewritten_query": "rewritten standalone searchable query (same as original_query if no change needed)"
        }}
    ]
}}

Example:
Original q2: "Who is its founder?", depends on q1, q1 answer is "OpenAI"
→ rewritten_query = "Who is the founder of OpenAI?"
""".strip()

RELEVANCE_AND_REWRITE_PROMPT = """
You are a rigorous retrieval evaluation expert. Given the user question and a batch of retrieved document snippets, complete two tasks:

1. Determine whether all retrieved documents collectively are sufficient to answer the user question completely and accurately
2. If insufficient, provide a more precise improved query that can increase recall (must include necessary context)

User question: {query}
Known preceding answer context (if any): {context_dependency}
Retrieved document snippets (sorted by relevance): {documents}

Respond strictly in the following JSON format (JSON only):
{{
    "is_relevant": true/false,
    "reason": "Brief reason, keep under 50 words",
    "improved_query": "",
    "needs_context": true/false
}}
""".strip()

GENERATE_ANSWER_PROMPT = """
You are a rigorous and accurate Q&A assistant. Answer the user question based ONLY on the provided context materials. Never fabricate information not present in the context.

Requirements:
1. Use only information from "previous answer context" and "current retrieval context"
2. Answer must be concise, direct, complete, and logically coherent
3. Output the answer content directly, without any prefix

User question: {query}
Previous answer context (answers to resolved dependency questions): {previous_answers}
Current retrieval context (reliable information retrieved this time): {context}

Provide the answer directly:
""".strip()

SELF_CHECK_PROMPT = """
You are a high-quality answer review expert. Strictly check whether the given answer is completely faithful to the provided documents, and whether there is any hallucination, omission of key information, or factual error.

User question (for reference only): {query}
Answer to be checked: {answer}
All document snippets used as basis for the answer:
{documents}

Output strictly in the following JSON format (JSON only):
{{
    "is_valid": true/false,
    "issues": "none" or brief description of problems,
    "revised_answer": ""
}}
""".strip()

SYNTHESIZE_ANSWERS_PROMPT = """
You are an answer synthesis expert. Your task is to give a concise, natural final answer to the original user question using the provided sub-answers as hidden reference only.

Original question: {original_query}

Reference sub-answers (do NOT mention them or quote them directly):
{sub_answers_with_dependencies}

CRITICAL RULES — follow exactly:
- Answer the original question directly in 1–4 sentences (preferably 1–2).
- Do NOT explain reasoning process.
- Do NOT mention “sub-questions”, “first”, “second”, “according to source X”.
- Do NOT repeat the question.
- Do NOT add meta-commentary like “Based on the analysis…”.
- Remove all redundancy and tracing of intermediate steps.
- If the answer naturally fits in one short paragraph, use only that.

Output the final answer ONLY — no JSON, no titles, no markdown, nothing else.
""".strip()
