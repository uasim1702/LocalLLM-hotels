"""
This file builds or loads a local Chroma vector database that stores the embeddings
of hotel reviews from the TripAdvisor CSV file. It exposes a retriever object that
is later used by main.py to fetch the most relevant reviews for any given question.

Behavior:
- On first run (when no database exists):
  1. Reads all reviews from the CSV file.
  2. Splits the text into smaller chunks for better retrieval.
  3. Creates vector embeddings using Ollama’s model ("mxbai-embed-large").
  4. Adds these vectors to a local Chroma database in batches of 1000.
  5. Prints a simple progress line (so the user knows it’s working).
- On later runs (when the DB already exists):
    Loads instantly.
Exports:
- retriever: A LangChain retriever configured to return the top 5 most relevant
  review chunks for each question. (You can change k=5 → k=10 for broader results.)
"""
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pandas as pd

DB_DIR = "./chroma_langchain_db"
COLLECTION = "tripadvisor_hotel_reviews"
CSV_PATH = "tripadvisor_hotel_reviews.csv"

EMB = OllamaEmbeddings(model="mxbai-embed-large")

def _build_or_load_store() -> Chroma:
    """
    Opens an existing Chroma database if found,
    or builds a new one from the CSV file if it doesn’t exist.

    Returns:
        A Chroma vector store ready for use as a retriever.
    """ 
    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=DB_DIR,
        embedding_function=EMB
    )

    # If store is empty, populate it from CSV
    is_empty = (not os.path.exists(DB_DIR)) or (store._collection.count() == 0)
    if is_empty:
        df = pd.read_csv(CSV_PATH).fillna("")
        # The splitter cuts each review into 600-character chunks with 100-character overlap.
        # This improves retrieval accuracy because the model works better with smaller,
        # focused pieces of text.
        #
        # You can tune these values:
        # chunk_size = 600 to 800 = larger context (slower, bigger DB)
        # chunk_size = 400  smaller context (faster, more precise)
        # chunk_overlap = 100 to 50 = less duplication, faster
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        
        docs, ids = [], []
        
        # Convert reviews to Document objects
        for i, row in df.iterrows():
            review = str(row.get("Review", "")).strip()
            if not review:
                continue
            rating = str(row.get("Rating", ""))
            for j, chunk in enumerate(splitter.split_text(review)):
                docs.append(Document(page_content=chunk, metadata={"rating": rating}))
                ids.append(f"{i}_{j}") # Unique ID for each document

        # BATCH controls how many documents are processed at once.
        # Larger batch - 1000: faster but uses more RAM.
        # Smaller batch - 500: slower but safer on lower memory.
        BATCH = 1000
        total = len(docs)
        
        for s in range(0, total, BATCH):
            e = min(s + BATCH, total)
            store.add_documents(docs[s:e], ids=ids[s:e])
            print(f" added: {e}/{total}") # progress line for updates on each batch

        # adds the data to a disk so we can use it later
        store.persist()

    return store

# Build/open the store once and expose a retriever
_store = _build_or_load_store()
retriever = _store.as_retriever(search_kwargs={"k": 5}) # return top 5 relevant chunks for each question.
# We can change k=5 to k=10 for broader coverage or to capture more context.
# However, more chunks = longer prompts and possibly slower answers.
