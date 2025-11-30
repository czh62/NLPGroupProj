## Usage

### Key Scripts: generateAnswers.py and app.py

#### generateAnswers.py
This script performs batch processing on the test set to generate answers using hybrid retrieval (BM25 + BGE) followed by reranking and LLM generation. It saves results in JSONL format for evaluation. Useful for final performance submission and testing.

- **Purpose**: Automate answer generation for the entire test split (1,052 queries). Outputs include query ID, generated answer, and retrieved document IDs with scores.
- **How to Run**:
  1. Ensure indexes are built (e.g., BM25 and BGE indexes in `data/`).
  2. Update `config.py` with paths and API keys.
  3. Run the script:
     ```
     python generateAnswers.py
     ```
  - It loads the test set, processes queries concurrently (with configurable threads), and saves output to `data/final_answers.jsonl` (configurable in `config.py`).
  - Example output format (per line):
    ```json
    {"id": "query_id", "text": "query_text", "answer": "generated_answer", "retrieved_docs": [["doc_id1", "score1"], ["doc_id2", "score2"]]}
    ```
- **Customization**: Adjust `max_workers` in ThreadPoolExecutor for parallelism. Modify retrievers or prompts in the script for experimentation.
- **Evaluation**: After running, use `eval_hotpotqa.py` or `eval_retrieval.py` on the output file.

#### app.py
This is the main user interface implemented as a Streamlit web app. It allows interactive querying with configurable retrieval modes, displays intermediate steps (e.g., decomposition, retrieval results, self-checking), and generates final answers.

- **Purpose**: Demonstrate the full RAG system in real-time, including multi-hop reasoning, hybrid retrieval, and agentic features. Ideal for demo videos and testing.
- **How to Run**:
  1. Ensure dependencies (including Streamlit) are installed.
  2. Update `config.py` with paths and API keys.
  3. Launch the app:
     ```
     streamlit run app.py
     ```
  - Access the app in your browser (default: http://localhost:8501).
- **Interface Features**:
  - **Sidebar Configuration**: Choose "Single Retrieval" (one method) or "Hybrid Retrieval" (two methods + reranking). Select from available retrievers (e.g., BM25, BGE).
  - **Query Input**: Enter a question (e.g., multi-hop example provided).
  - **Tabs for Workflow**:
    - Query Decomposition: Breaks complex queries into sub-queries with dependencies.
    - Sub-query Processing: Shows rewriting, iterative retrieval (up to 3 attempts with relevance checks), document scores, and generated sub-answers.
    - Final Synthesis: Combines sub-answers into a cohesive final response.
  - **Output**: Final answer with copyable text area. Supports multi-turn implicitly via context management.
- **Tips**: For multi-turn, enter follow-up queries referencing previous ones. The app handles dependencies automatically.

### Other Usage Notes
- **Building Indexes**: Run individual retriever scripts (e.g., `BM25Retriever.py`) to build indexes if not present.