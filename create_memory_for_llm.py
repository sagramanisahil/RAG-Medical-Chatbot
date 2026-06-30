import os
import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DATA_PATH = "data/"

def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyMuPDFLoader)
    
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
print("Length of PDF is:", len(documents))

# NEW STEP — merge pages per source file into one big Document
from langchain_core.documents import Document
from collections import defaultdict

def merge_pages_by_source(documents):
    grouped = defaultdict(list)
    for doc in documents:
        grouped[doc.metadata['source']].append(doc)
    
    merged_docs = []
    for source, docs in grouped.items():
        docs_sorted = sorted(docs, key=lambda d: d.metadata.get('page', 0))
        full_text = "\n".join(d.page_content for d in docs_sorted)
        merged_docs.append(Document(page_content=full_text, metadata={'source': source}))
    return merged_docs

documents = merge_pages_by_source(documents)
print("Merged into", len(documents), "documents (one per source file)")



def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=300)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print("Length of Text Chunks: ", len(text_chunks))


def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()


DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)