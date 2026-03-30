import asyncio
import os
import ssl
import certifi

# Ignore warnings
import os
import transformers
import warnings
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

from typing import List, Any, Dict
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_info, log_success, log_error, log_warning, log_header)
load_dotenv()

# Configure SSL context to use certifi's CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Set HuggingFace token for embeddings
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = PineconeVectorStore(
    index_name=os.getenv("DOCUMENT_INDEX_NAME"),
    embedding=embeddings,
)

tavily_crawl = TavilyCrawl()
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)

async def index_documents(docs: List[Document], batch_size: int = 50):
    "Process documents in batches asynchronously"
    log_header("Vector Storage Phase")
    log_info(f"VectorStore: Preparing to add {len(docs)} documents to the vector store", Colors.DARKCYAN)

    # Create batches of documents
    batches = [
        docs[i : i + batch_size] for i in range(0, len(docs), batch_size)
    ]
    log_info(f"VectorStore: Split documents into {len(batches)} batches of size {batch_size} documents each.")

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            vectorstore.add_documents(batch)
            log_success(f"VectorStore: Successfully added batch {batch_num}/{len(batches)} with {len(batch)} documents.")
        except Exception as e:
            log_error(f"VectorStore: Error adding batch {batch_num}/{len(batches)} - {str(e)}")
            return False
        return True
    
    tasks = [add_batch(batch, idx + 1) for idx, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )

async def main():
    "Main function started"
    log_header("Document Ingestion Pipeline with Tavily and Pinecone")

    log_info(
        "TavilyCrawl: Starting to crawl the documentation from https://python.langchain.com/",
        Colors.PURPLE,
    )
    # Crawl the documentation site

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 5,
            "extract_depth": "advanced",
            # "instructions": "content on ai agents" -- we can use this to focus the crawl on specific topics, but for now we'll get everything and filter later
        }
    )

    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) for result in res["results"]]
    log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site.")

    #Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"TextSplitter: Successfully split documents into {len(splitted_docs)}) chunks.")

    # Process documents asynchronously
    await index_documents(splitted_docs, batch_size=50)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())