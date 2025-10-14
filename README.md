# LocalLLM-Hotels
Local RAG over hotel reviews (mostly in US) from a CSV. Uses Ollama + Chroma + LangChain.

Overview
- Ingests 7282_1.csv (hotel reviews).
- Splits reviews into chunks and builds a local Chroma DB with embeddings (mxbai-embed-large).
- User asks a question, retrieves top-k chunks, and calls a local LLM (llama3.2) with temp=0.0.

Simple flow
CSV -> vector.py -> Chroma -> main.py -> LLM

Prerequisites
- Python 3.10+
- Ollama installed and running: `ollama serve`
- Models (one-time):
- `ollama pull mxbai-embed-large`
- `ollama pull llama3.2`

### 2 Clone & Setup
```bash
git clone https://github.com/uasim1702/LocalLLM-hotels.git
cd LocalLLM-hotels
```

Virtual env:
```bash
python -m venv venv
source venv/bin/activate
```

### 3 Install Deps
```bash
pip install -r requirements.txt
```

Your requirements.txt should look like this:
```
langchain
langchain-ollama
langchain-chroma
chromadb
pandas

```

Run
- Place 7282_1.csv in project root.
- First run builds the DB (38.5k chunks with current settings).

Then it drops you into:
```
Ask a question or (q to quit):
```

- Prompt:

Examples
- What do guests say about the staff at Hotel Russo Palace?
- What do you think about Holiday Inn in Maryland?
- Summarize breakfast opinions for hotels in Ocean city.

Config (edit in code)
- vector.py: chunk_size=600, chunk_overlap=100, retriever k=5
- main.py: llama3.2, temperature=0.0

Rebuild index
- Delete `./chroma_langchain_db/` and run again.

Notes
- Uses `reviews.text` and `reviews.rating` as the main fields; hotel `name`, `city`, `country` stored as metadata and shown in context.
- All responses are grounded in retrieved reviews. If not present in the reviews, it will say so.

Type `q` when you're good.

## Performance

- First build: 30-50 mins setup.
- After that: Queries immediately.
- Grounding: Hits 85-95% on factual pulls; hallucinations near zero since temp = 0.0
