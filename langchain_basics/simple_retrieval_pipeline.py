from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
load_dotenv()

# Ignore warnings
import os
import transformers
import warnings
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Set HuggingFace token for embeddings
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings

##============================================================
# STEP 1 - Initialize the embeddings, vector store and LLM
#============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    temperature=1.0
    )

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings,
)

#============================================================
# STEP 2 - Load and split documents, then add to vector store
#============================================================

loader = TextLoader(os.getenv("FILE_PATH"))  # A text file with some content about Pinecone and RAG
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splitted_docs = text_splitter.split_documents(documents)

PineconeVectorStore.from_documents(
    documents=splitted_docs,
    embedding=embeddings,
    index_name=os.getenv("INDEX_NAME")
)

#============================================================
# STEP 3 - Create a retriever and format docs
#============================================================

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    """Format retrieved documents into a singlestring"""
    return "\n\n".join(doc.page_content for doc in docs)

#============================================================
# STEP 4 - Test the retrieval chain without LCEL
#============================================================

def retrieval_chain_without_lcel(query: str):
    """A simple retrieval chain without LCE or tool calls"""
    # Step 1 - Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)

    # Step 2 - Format retrieved documents into a single string
    context = format_docs(retrieved_docs)

    # Step 3 - Create a prompt with the context and question
    prompt = f"""Answer the question based only on the following
     context: {context}
     Question: {query}
     Provide a detailed answer:"""

    # Step 4 - Invoke the LLM with the prompt
    result = llm.invoke([HumanMessage(content=prompt)])

    return result.content

#============================================================
# STEP 5 - Test the retrieval chain with LCEL
#============================================================
def retrieval_chain_with_lcel(query: str):
    """A retrieval chain with LCEL and tool calls"""
    # Step 1: Define prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the question using only the context below.

    Context: {context}
    Question: {question}
    """)

    # Step 2: Build chain using pipe operator
    chain = (
        {
            "context": retriever | format_docs,          # retriever fetches chunks
            "question": RunnablePassthrough()  # passes question as-is
        }
        | prompt          # chunks + question → fills the prompt template
        | llm             # filled prompt → sent to LLM
        | StrOutputParser()  # LLM response → plain string
    )

    # Step 3: Invoke the chain with the question
    return chain.invoke(query)

query = "What is pinecone in Machine Learning?"

##===========================================================
# RAW INVOCATION WITHOUT RAG
##============================================================
print("\n" + "="*20 + " RAW INVOCATION WITHOUT RAG " + "="*20)
result_raw = llm.invoke([HumanMessage(content=query)])
print("\n Answer:")
print(result_raw.content)

##===========================================================
# RETRIEVAL CHAIN WITHOUT LCE
##============================================================
print("\n" + "="*20 + " RETRIEVAL CHAIN WITHOUT LCE " + "="*20)
result_retrieval = retrieval_chain_without_lcel(query)
print("\n Answer:")
print(result_retrieval)

#============================================================
# RETRIEVAL CHAIN WITH LCE
##============================================================
print("\n" + "="*20 + " RETRIEVAL CHAIN WITH LCE " + "="*20)
result_retrieval_lcel = retrieval_chain_with_lcel(query)
print("\n Answer:")
print(result_retrieval_lcel)