# Local LLM + RAG on TripAdvisor Reviews

Ask hotel questions using a **local** LLM with review-based retrieval.

## Stack
- **LLM:** `llama3.2` via Ollama  
- **Embeddings:** `mxbai-embed-large` via Ollama  
- **Vector DB:** Chroma (persisted in `./chroma_langchain_db`)  
- **Orchestration:** LangChain

## Setup
```bash
ollama pull llama3.2
ollama pull mxbai-embed-large

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Data
Place tripadvisor_hotel_reviews.csv in the project root.
Expected columns: Review, Rating, optional Date.

Run
python3 main.py
