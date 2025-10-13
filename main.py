from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vector import retriever


"""
You can swap models here:
Change "llama3.2" to another local model you've pulled with Ollama.
TEMPERATURE controls creativity or strict to the csv:
0.0 = No hallucination.
0.7 = more creative generates answers on its own."""

model = OllamaLLM(model="llama3.2", temperature=0.0)

PROMPT = """
You are an assistant that answers ONLY using the reviews below.
- When possible, explicitly mention the hotel name and its area (city, country).
- If something isn’t present in the reviews, say: “Not mentioned in the reviews.”

# Reviews
{reviews}

# Question
{question}
"""

prompt = ChatPromptTemplate.from_template(PROMPT)
parser = StrOutputParser()

def format_docs(docs):
    """
    Compact, numbered text block with helpful metadata.
    Truncates long reviews to 450 chars.
    """
    lines = []
    for d in docs:
        text = d.page_content.strip().replace("\n", " ")
        if len(text) > 450:
            text = text[:450] + "..."

        m = d.metadata or {}
        name     = m.get("name", "")
        city     = m.get("city", "")
        country  = m.get("country", "")
        rating   = m.get("rating", "")
        rdate    = m.get("review_date", "")

        hotel_info = []
        if name: hotel_info.append(name)
        area = ", ".join([x for x in [city, country] if x])
        if area: hotel_info.append(f"({area})")
        header = " ".join(hotel_info).strip() or "Unknown hotel"

        tail = []
        if rating: tail.append(f"rating: {rating}")
        if rdate:  tail.append(f"date: {rdate}")
        meta_tail = " ".join(tail)

        if meta_tail:
            lines.append(f"- {header} — {meta_tail}\n  {text}")
        else:
            lines.append(f"- {header}\n  {text}")
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
        print("\nThinking...\n")
        print("\nAnswer:")
        print(rag_chain.invoke(q))
        print("--------------\n")
