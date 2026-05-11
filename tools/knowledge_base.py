import requests
from bs4 import BeautifulSoup
from sec_api import QueryApi
from dotenv import load_dotenv
import os
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List , Dict , Any 
load_dotenv()


PERSIST_DIR = "../chroma_db"

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding
)


def ticker_exists(ticker: str) -> bool:
    results = vectordb._collection.get(
        where={"ticker": ticker},
        limit=1
    )
    return len(results["ids"]) > 0


def fetch_sec_filing(ticker: str, report_type: str) -> str:
    api_key = os.getenv("SEC_API_KEY")
    queryApi = QueryApi(api_key=api_key)

    query = {
        "query": {
            "query_string": {
                "query": f"ticker:{ticker} AND formType:\"{report_type}\""
            }
        },
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}]
    }

    filings = queryApi.get_filings(query)

    if not filings["filings"]:
        return "no data available"

    filing = filings["filings"][0]
    return filing["linkToFilingDetails"]


def fetch_clean_10k_sections(url: str) -> dict:
    headers = {"User-Agent": "your.email@example.com"}
    if url is "no data available":
        return {}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text(separator="\n")

    text = re.sub(r"\n\s*\n", "\n\n", text)

    patterns = {
        "business": r"Item\s+1\.\s+Business(.*?)(Item\s+1A\.|Item\s+2\.)",
        "risk_factors": r"Item\s+1A\.\s+Risk\s+Factors(.*?)(Item\s+1B\.|Item\s+2\.)",
        "mda": r"Item\s+7\.\s+Management.*Discussion.*Analysis(.*?)(Item\s+7A\.|Item\s+8\.)",
    }

    sections = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()

    return sections



def store_in_chroma(ticker: str, sections: dict):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = []
    for section_name, content in sections.items():
        chunks = splitter.split_text(content)

        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "ticker": ticker,
                    "section": section_name
                }
            ))

    if docs:

        vectordb.add_documents(docs)

        print(f"{ticker} stored in DB")


def ensure_knowledge_base(ticker: str):
    if ticker_exists(ticker):
        print(f"{ticker} already exists in DB ")
    else:
        print(f"{ticker} not found → fetching...")
        url = fetch_sec_filing(ticker, "10-K")
        sections = fetch_clean_10k_sections(url)
        if sections:
            store_in_chroma(ticker , sections)


def query_rag(ticker: str, query: str) -> List[Dict[str, Any]]:
    ensure_knowledge_base(ticker)
    retriever = vectordb.as_retriever(
            search_type = "mmr",
            search_kwargs = {"k": 5, "fetch_k": 20 , "filter": {"ticker": ticker}}
    )

    docs = retriever.invoke(query)

    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    return results