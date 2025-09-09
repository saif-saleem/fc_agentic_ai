import os
import re
import shutil
import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
# === Load env variables ===
load_dotenv()

# === Constants ===
BASE_DIR = "app/data/GS"   # Folder with your GS PDFs/docs
CHROMA_DIR = "app/embeddings_gs/all_documents"  # Where Chroma DB will be stored
TOKEN_LIMIT = 280_000
MAX_TOKENS_PER_CHUNK = 8191
CHUNK_SIZE = 1000            # Smaller chunks for better matching
CHUNK_OVERLAP = 150
MODEL_NAME = "text-embedding-ada-002"
COLLECTION_NAME = "gs_all_documents"

encoding = tiktoken.encoding_for_model(MODEL_NAME)

# === Helper Functions ===
def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def safe_filter_metadata(meta: dict) -> dict:
    """
    Keep all useful metadata fields so retrieval filters work properly.
    """
    allowed_keys = {
        "source", "page", "clause", "project_id",
        "document_type", "project_type", "project_title", "subfolder"
    }
    return {
        k: str(v)[:200]  # limit to avoid very large metadata
        for k, v in meta.items()
        if k in allowed_keys and isinstance(v, (str, int, float))
    }

def extract_clause_number(text: str) -> str:
    match = re.search(r'Clause\s+([\d\.]+)', text, re.IGNORECASE)
    return match.group(1) if match else "N/A"

def get_loader(fname: str, path: str):
    """
    Return correct loader based on file extension.
    """
    if fname.lower().endswith(".pdf"):
        return PDFPlumberLoader(path)
    elif fname.lower().endswith((".docx", ".doc")):
        return UnstructuredWordDocumentLoader(path)
    elif fname.lower().endswith((".xls", ".xlsx")):
        return UnstructuredExcelLoader(path, mode="elements")
    return None

# === Unified Document Loader ===
def load_all_documents():
    all_docs = []
    folders = [("standard", "Standard_documents"), ("project", "Project_documents")]

    print("📥 Scanning all documents for loading...")
    for doc_type, folder_name in folders:
        folder_path = os.path.join(BASE_DIR, folder_name)
        for root, _, files in os.walk(folder_path):
            for fname in tqdm(files, desc=f"🔍 Processing files in {folder_name}"):
                path = os.path.join(root, fname)
                loader = get_loader(fname, path)
                if not loader:
                    print(f"⏭️ Skipped unsupported file: {fname}")
                    continue

                try:
                    loaded = loader.load()
                    rel_path = os.path.relpath(root, BASE_DIR)

                    project_id = "STANDARD" if doc_type == "standard" else (
                        re.search(r'\(ID\s*(\d+)\)', rel_path, re.IGNORECASE).group(1)
                        if re.search(r'\(ID\s*(\d+)\)', rel_path, re.IGNORECASE) else "UNKNOWN"
                    )

                    for i, doc in enumerate(loaded):
                        doc.metadata.update({
                            "source": fname,
                            "page": i + 1,
                            "clause": extract_clause_number(doc.page_content),
                            "project_type": "GS",
                            "project_id": project_id,
                            "project_title": os.path.basename(root),
                            "document_type": doc_type,
                            "subfolder": rel_path
                        })
                    all_docs.extend(loaded)

                except Exception as e:
                    print(f"❌ Error processing {fname}: {e}")

    return all_docs

# === Chroma Add and Persist ===
def add_docs_to_chroma(docs, persist_path):
    if not docs:
        print(f"⚠️ No documents to index at {persist_path}.")
        return

    # Always start fresh to avoid inconsistent metadata
    if os.path.exists(persist_path):
        print("🧹 Removing existing Chroma DB...")
        shutil.rmtree(persist_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    print("🔪 Splitting documents into chunks...")
    chunks = splitter.split_documents(docs)
    print(f"📄 Total Chunks: {len(chunks)}")

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        chunk_size=100
    )

    db = Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    batch, tokens, batch_id = [], 0, 1

    print("💾 Adding chunks to Chroma with progress...")
    for doc in tqdm(chunks, desc="📦 Embedding batches"):
        token_len = count_tokens(doc.page_content)

        if token_len == 0 or token_len > MAX_TOKENS_PER_CHUNK:
            continue

        if tokens + token_len > TOKEN_LIMIT:
            db.add_documents([
                Document(page_content=d.page_content, metadata=safe_filter_metadata(d.metadata))
                for d in batch
            ])
            print(f"✅ Added batch {batch_id} ({len(batch)} docs)")
            batch, tokens, batch_id = [], 0, batch_id + 1

        batch.append(doc)
        tokens += token_len

    # Add remaining batch
    if batch:
        db.add_documents([
            Document(page_content=d.page_content, metadata=safe_filter_metadata(d.metadata))
            for d in batch
        ])
        print(f"✅ Final batch {batch_id} added.")

    db.persist()
    print(f"✅ Chroma DB stored at: {persist_path}")

# === Entry Point ===
def create_chroma_index():
    print("🚀 Starting Chroma Index Creation Pipeline...\n")
    print("📥 Loading all documents (standard + project)...")
    all_docs = load_all_documents()

    print("\n💾 Storing all documents in unified Chroma DB...")
    add_docs_to_chroma(all_docs, CHROMA_DIR)

    print("\n🎉 All done! Chroma index created successfully.")

if __name__ == "__main__":
    create_chroma_index()
