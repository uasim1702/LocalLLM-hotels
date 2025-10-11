"""
Console RAG app that answers questions only using hotel reviews retrieved
from a local Chroma vector store. The retriever is provided by vector.py.

Usage:
    (venv) python main.py
    Ask (q to quit): <your question>
"""
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vector import retriever

"""
You can swap models here:
Change "llama3.2" to another local model you've pulled with Ollama, e.g. "llama3.1" or "qwen2".
TEMPERATURE controls creativity or strict to the csv:
0.0 = factual and consistent no hallucination
0.7 = more creative genereates answers on its own."""

model = OllamaLLM(model="llama3.2", temperature=0.0)

PROMPT = """
You are an assistant that answers ONLY using the reviews below.
If the reviews don’t mention something, say: “Not mentioned in the reviews."

# Reviews
{reviews}

# Question
{question}
"""

prompt = ChatPromptTemplate.from_template(PROMPT)
parser = StrOutputParser()

def format_docs(docs):
    """
    Convert retrieved review documents into a compact, numbered text block
    suitable for prompting.

    - Truncates long reviews to 450 chars.
    """
    lines = []
    for d in docs:
        text = d.page_content.strip().replace("\n", " ")
        if len(text) > 450:
            text = text[:450] + "..."
        lines.append(f"- {text}")
    return "\n".join(lines)

# RAG pipeline:
# 1) Take user question
# 2) Retrieve top-k reviews
# 3) Format them into the prompt
# 4) Generate model answer
# 5) Parse to string

rag_chain = (
    {"question": RunnablePassthrough(), "reviews": retriever | format_docs}
    | prompt
    | model
    | parser
)

if __name__ == "__main__":
    while True:
        q = input("Ask a question or (q to quit): ").strip()
        if q.lower() == "q":
            break
        print("\Thinking...\n")
        print("\nAnswer:")
        print(rag_chain.invoke(q))
        print("--------------\n")
