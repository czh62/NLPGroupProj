# Prompt for determining if a query needs decomposition and generating sub-queries if necessary.
DECOMPOSITION_PROMPT = """
You are an expert query analyzer. Determine if the following query is complex and requires decomposition into sub-queries for better retrieval-augmented generation. Complex queries are multi-hop, involve multiple entities, or require reasoning across steps.

Query: {query}

If the query is simple (single-hop, direct fact), respond with: "No decomposition needed."

If decomposition is needed, list 2-4 sub-queries that break it down logically. Each sub-query should be self-contained and answerable independently. Output in JSON format:
{{
  "needs_decomposition": true,
  "sub_queries": ["sub_query1", "sub_query2", ...]
}}
If no decomposition: 
{{
  "needs_decomposition": false,
  "sub_queries": []
}}
"""

# Prompt for checking relevance of retrieved documents to the query.
RELEVANCE_CHECK_PROMPT = """
You are a relevance evaluator. Given the query and the retrieved documents, determine if the documents are relevant to answering the query. Relevant means the documents contain information directly related to the key elements of the query.

Query: {query}
Retrieved Documents:
{documents}

Respond with JSON:
{{
  "is_relevant": true/false,
  "reason": "brief explanation",
  "suggested_rewrite": "if not relevant, suggest a rewritten query, else empty string"
}}
"""

# Prompt for rewriting the query if relevance check fails.
QUERY_REWRITE_PROMPT = """
You are a query rewriter. Rewrite the original query to improve retrieval results, based on the suggestion and previous failure reason. Make it more precise, add synonyms, or rephrase for clarity.

Original Query: {original_query}
Failure Reason: {reason}
Suggested Rewrite: {suggested_rewrite}

Output the rewritten query as plain text.
"""

# Prompt for generating an answer based on retrieved documents.
GENERATE_ANSWER_PROMPT = """
You are a helpful assistant. Using ONLY the provided context from retrieved documents, answer the query accurately and concisely. If the context doesn't have enough information, say "Insufficient information in context."

Query: {query}
Context:
{context}

Output the answer in plain text, followed by a newline, then "Evidence: " and list the document IDs used.
"""

# Prompt for self-checking the generated answer against the evidence.
SELF_CHECK_PROMPT = """
You are a fact-checker. Verify if the generated answer is fully supported by the provided evidence without hallucinations. Check for accuracy, consistency, and completeness.

Generated Answer: {answer}
Evidence Documents:
{documents}

Respond with JSON:
{{
  "is_valid": true/false,
  "issues": "list any issues or empty string",
  "revised_answer": "if not valid, provide a revised answer, else empty string"
}}
"""

# Prompt for synthesizing multiple sub-answers into a final answer.
SYNTHESIZE_ANSWERS_PROMPT = """
You are a response synthesizer. Combine the following sub-answers into a coherent, complete final answer for the original query. Ensure logical flow and no contradictions.

Original Query: {original_query}
Sub-Answers:
{sub_answers}

Output the final answer in plain text.
"""