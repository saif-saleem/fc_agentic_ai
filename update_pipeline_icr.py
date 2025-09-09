import os, re
import tiktoken
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

# === Load environment variables ===
load_dotenv()

# === Constants ===
BASE_DIR = "app/data/ICR"
CHROMA_BASE_DIR = "app/embeddings_icr"
TOKEN_LIMIT = 280_000
MAX_TOKENS_PER_CHUNK = 8191
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MODEL_NAME = "text-embedding-ada-002"
encoding = tiktoken.encoding_for_model(MODEL_NAME)

def count_tokens(text): return len(encoding.encode(text))
def safe_filter_metadata(meta): return {k: v for k, v in meta.items() if not isinstance(v, (list, dict))}
def extract_clause_number(text): 
    match = re.search(r'Clause\s+([\d\.]+)', text, re.IGNORECASE)
    return match.group(1) if match else "N/A"

def get_loader(fname, path):
    if fname.endswith(".pdf"):
        return PDFPlumberLoader(path)
    elif fname.endswith((".docx", ".doc")):
        return UnstructuredWordDocumentLoader(path)
    elif fname.endswith((".xls", ".xlsx")):
        return UnstructuredExcelLoader(path, mode="elements")
    return None

def load_docs_from_subfolder(subfolder, doc_type):
    docs = []
    folder_path = os.path.join(BASE_DIR, subfolder)
    
    for root, _, files in os.walk(folder_path):
        for fname in files:
            path = os.path.join(root, fname)
            loader = get_loader(fname, path)
            if not loader:
                print(f"⏭️ Skipped unsupported file: {fname}")
                continue

            try:
                loaded = loader.load()
                rel_path = os.path.relpath(root, BASE_DIR)
                for i, doc in enumerate(loaded):
                    doc.metadata.update({
                        "source": fname,
                        "page": i + 1,
                        "clause": extract_clause_number(doc.page_content),
                        "project_type": "ICR",
                        "project_id": re.search(r'(ICR\d+)', fname, re.I).group(1) if re.search(r'(ICR\d+)', fname, re.I) else "ICR",
                        "project_title": os.path.splitext(fname)[0],
                        "document_type": doc_type,
                        "subfolder": rel_path
                    })
                docs.extend(loaded)
            except Exception as e:
                print(f"❌ Error processing {fname}: {e}")
    return docs

def add_to_chroma(docs, persist_path):
    if not docs:
        print(f"⚠️ No documents to add at {persist_path}")
        return
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"📄 Total Chunks for {persist_path}: {len(chunks)}")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), chunk_size=100)
    db = Chroma(persist_directory=persist_path, embedding_function=embeddings)

    batch, tokens, batch_id = [], 0, 1
    for doc in chunks:
        token_len = count_tokens(doc.page_content)
        if token_len == 0 or token_len > MAX_TOKENS_PER_CHUNK:
            continue

        if tokens + token_len > TOKEN_LIMIT:
            db.add_documents([
                Document(page_content=d.page_content, metadata=safe_filter_metadata(d.metadata)) 
                for d in batch
            ])
            print(f"✅ Added batch {batch_id} ({len(batch)} docs) to {persist_path}")
            batch, tokens, batch_id = [], 0, batch_id + 1

        batch.append(doc)
        tokens += token_len

    if batch:
        db.add_documents([
            Document(page_content=d.page_content, metadata=safe_filter_metadata(d.metadata)) 
            for d in batch
        ])
        print(f"✅ Final batch {batch_id} added to {persist_path}")

    db.persist()
    print(f"✅ Chroma DB stored at: {persist_path}")

def create_chroma_index():
    print("🔄 Loading standard documents...")
    standard_docs = load_docs_from_subfolder("Standard_documents", "standard")
    print("🔄 Loading project documents...")
    project_docs = load_docs_from_subfolder("Project_documents", "project")

    print("\n🧠 Creating vectorstore for STANDARD documents...")
    add_to_chroma(standard_docs, os.path.join(CHROMA_BASE_DIR, "standard_documents"))

    print("\n🧠 Creating vectorstore for PROJECT documents...")
    add_to_chroma(project_docs, os.path.join(CHROMA_BASE_DIR, "project_documents"))

if __name__ == "__main__":
    create_chroma_index()
