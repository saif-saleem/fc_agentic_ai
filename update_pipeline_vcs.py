import os, re
import tiktoken
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

load_dotenv()

BASE_DIR = "app/data/VERRA_VCS"
CHROMA_DIR = "app/embeddings_vcs"
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

def load_docs_from_subfolder(folder_name, doc_type):
    docs = []
    folder_path = os.path.join(BASE_DIR, folder_name)
    for root, _, files in os.walk(folder_path):
        for fname in files:
            path = os.path.join(root, fname)
            loader = get_loader(fname, path)
            if not loader:
                print(f"⏭️ Skipped unsupported file: {fname}")
                continue
            try:
                loaded = loader.load()
                for i, doc in enumerate(loaded):
                    doc.metadata.update({
                        "source": fname,
                        "page": i + 1,
                        "clause": extract_clause_number(doc.page_content),
                        "project_type": "VCS",
                        "project_id": re.search(r'(VCS\d+)', fname, re.I).group(1) if re.search(r'(VCS\d+)', fname, re.I) else "VCS",
                        "project_title": os.path.splitext(fname)[0],
                        "document_type": doc_type
                    })
                docs.extend(loaded)
            except Exception as e:
                print(f"❌ Error processing {fname}: {e}")
    return docs

def create_index(documents, persist_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"📄 Total Chunks for {persist_path}: {len(chunks)}")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), chunk_size=100)
    db = Chroma(persist_directory=persist_path, embedding_function=embeddings)

    batch, tokens, batch_id = [], 0, 1
    for doc in chunks:
        token_len = count_tokens(doc.page_content)
        if token_len == 0 or token_len > MAX_TOKENS_PER_CHUNK:
            continue

        if tokens + token_len > TOKEN_LIMIT:
            db.add_documents([Document(page_content=d.page_content, metadata=safe_filter_metadata(d.metadata)) for d in batch])
            print(f"✅ Added batch {batch_id} ({len(batch)} docs)")
            batch, tokens, batch_id = [], 0, batch_id + 1

        batch.append(doc)
        tokens += token_len

    if batch:
        db.add_documents([Document(page_content=d.page_content, metadata=safe_filter_metadata(d.metadata)) for d in batch])
        print(f"✅ Final batch {batch_id} added.")

    db.persist()
    print(f"✅ Stored ChromaDB at: {persist_path}")

def create_chroma_index():
    print("📁 Loading Standard Documents...")
    standard_docs = load_docs_from_subfolder("Standard_documents", "standard")
    print("📁 Loading Project Documents...")
    project_docs = load_docs_from_subfolder("Project_documents", "project")

    if not standard_docs and not project_docs:
        print("⚠️ No documents loaded.")
        return

    if standard_docs:
        create_index(standard_docs, os.path.join(CHROMA_DIR, "standard_documents"))
    if project_docs:
        create_index(project_docs, os.path.join(CHROMA_DIR, "project_documents"))

if __name__ == "__main__":
    create_chroma_index()
