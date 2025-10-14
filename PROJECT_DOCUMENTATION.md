# LocalLLM-Hotels — Retrieval-Augmented Generation on a local CSV

- Author: Uasim Hallak
- Date: October 2025

# Note:
Most hotels in this dataset are located in the United States, with a few entries from Europe.  
The data comes from the Datafiniti Hotel Reviews CSV (7282_1.csv), which contains real TripAdvisor-style reviews with hotel names and locations.

1) What this is:
- A small RAG setup that answers questions about hotel reviews from a local CSV.
- No internet. Local embeddings + local LLM via Ollama.

2) Data:
- CSV: 7282_1.csv
- Key fields used:
- reviews.text   -> review body
- reviews.rating -> rating value
- name           -> hotel name
- city, country  -> area info
- Other fields exist (address, province, postalCode, latitude, longitude, categories, reviews.date, reviews.username). They are stored as metadata when available.

3) How it works:
- CSV -> vector.py -> (split into chunks) -> embeddings (Ollama mxbai-embed-large) -> Chroma DB (local) -> main.py -> retrieve top-k chunks -> prompt LLM (llama3.2, temp=0.0) -> answer grounded in reviews.

4) Notes:
- Current split produced ~38.5k chunks (chunk_size=600, overlap=100).
- First run builds the DB and saves it in ./chroma_langchain_db/.
- Next runs reuse the DB.

5) Important parameters (tune if needed):
- chunk_size    (vector.py) default 600
- chunk_overlap (vector.py) default 100
- k             (vector.py) default 5 (top-k retrieved chunks)
- temperature   (main.py)   default 0.0

6) Rebuilding the DB:
If the CSV changes or you want a fresh index:
- delete ./chroma_langchain_db/
- run main.py again (vector.py will rebuild automatically on first use)

7) Assumptions and limits
- Answers are limited to what is in reviews.text. If not present, the app says: "Not mentioned in the reviews."
- Dataset contains mixed encodings in some rows. Loaded with UTF-8 using pandas.

8) Tools and learning sources
I built this project by combining several learning resources:
- Watched those YouTube tutorials: https://www.youtube.com/watch?v=pTaSDVz0gok, https://www.youtube.com/watch?v=osKyvYJ3PRM&t=1136s, 
- Read the book *“Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow”* about a month ago, which helped me understand the basics of machine learning and training concepts.  
- Used ChatGPT for most of the documentation writing, mistake explanations, and general guidance while developing the code.

10) Quick maintenance notes
- To change how many results feed the prompt, set k in vector.py retriever.
- To change LLM or embedding model, edit main.py (model name) and vector.py (embedding model).
- To add more metadata into the prompt, adjust format_docs() in main.py.
