from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# 1. Load PDF
loader = PyPDFLoader("Concept Note.pdf")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 3. Embeddings + Vector DB
emb = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, emb)

# 4. Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# 5. Prompt template for RAG
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
)

# 6. LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# 7. Build the RAG pipeline using Runnables (new API!)
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | RunnableLambda(format_docs),
     "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 8. Ask a question
query = "What is the main purpose of the Concept Note?"
response = rag_chain.invoke(query)

print("\n===== ANSWER =====\n")
print(response.content)

