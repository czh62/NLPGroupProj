DECOMPOSITION_PROMPT = """
You are a professional query decomposition expert. Your task is to determine whether the user's question is complex enough to require decomposition into multiple sub-questions.

Rules:
- If the question is simple, straightforward, or can be answered directly without additional breakdown, return needs_decomposition = false.
- If the question involves multiple entities, multiple conditions, comparisons, causal reasoning, multi-step logic, or timeline analysis, it likely needs decomposition.
- Only decompose when decomposition will improve retrieval or answer quality. Do NOT decompose simple or single-step questions.
- Sub-questions must be complete, clear, and independently understandable. 
- Sub-questions may be independent or dependent on earlier ones (use depends_on for dependent questions).
- Recommended number of sub-queries: 2–10, with a maximum of 10.

User question: {query}

Respond strictly in JSON format (no additional explanations):
{{
    "needs_decomposition": true/false,
    "sub_queries": [
        {{
            "query": "Sub-question 1 (clear and self-contained)",
            "id": "Q1",
            "depends_on": []
        }},
        {{
            "query": "Sub-question 2",
            "id": "Q2",
            "depends_on": ["Q1"]
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
You are an advanced answer synthesis expert. The user has asked a complex question, and the system has provided multiple sub-question answers as reference information.

Your task is to produce a complete, natural, and coherent final answer that responds directly to the original question.

Original question: {original_query}

Reference information (answers to decomposed sub-questions):
{sub_answers_with_dependencies}

Requirements:
- Use the sub-question answers only as reference; do not mention or refer to the sub-questions in the final response
- Provide only the final answer to the original question, without explaining the process or referencing the source information
- Ensure the language is smooth, logically clear, and free from unnecessary repetition
- Output only the final answer—no titles, explanations, or citation markers

Now generate the final answer:
""".strip()
