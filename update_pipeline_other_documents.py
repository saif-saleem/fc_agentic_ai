import os, re, traceback
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import tiktoken

# 🔧 Load environment variables
load_dotenv()

# 📁 Directory Configs
BASE_DIR = "app/data/OTHER_DOCUMENTS"
CHROMA_BASE_DIR = "app/embeddings_other_documents"
VALID_SUBTYPES = ["carbon_market_general_document", "IPCC"]
TOKEN_LIMIT = 280_000
MAX_TOKENS_PER_CHUNK = 8191

# 🧠 Tokenizer for token-based batching
encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def safe_filter_metadata(meta: dict) -> dict:
    """Remove complex objects from metadata."""
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}

def extract_clause_number(text: str) -> str:
    match = re.search(r'Clause\s+([\d\.]+)', text, re.IGNORECASE)
    return match.group(1) if match else "N/A"

def load_file(path: str):
    """Auto-select loader based on file extension."""
    if path.endswith(".pdf"):
        return PDFPlumberLoader(path)
    elif path.endswith((".doc", ".docx")):
        return UnstructuredWordDocumentLoader(path)
    elif path.endswith((".xls", ".xlsx")):
        return UnstructuredExcelLoader(path, mode="elements")
    return None

def create_chroma_index():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), chunk_size=100)

    for subtype in VALID_SUBTYPES:
        print(f"\n🔍 Processing subtype: {subtype}")
        folder_path = os.path.join(BASE_DIR, subtype)
        chroma_path = os.path.join(CHROMA_BASE_DIR, subtype)

        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        docs = []
        for root, _, files in os.walk(folder_path):
            for fname in files:
                file_path = os.path.join(root, fname)
                loader = load_file(file_path)
                if not loader:
                    continue
                try:
                    loaded_docs = loader.load()
                    for i, doc in enumerate(loaded_docs):
                        doc.metadata.update({
                            "source": fname,
                            "page": i + 1,
                            "clause": extract_clause_number(doc.page_content),
                            "subtype": subtype,
                            "file_path": file_path,
                            "project_id": os.path.splitext(fname)[0],
                            "project_title": os.path.splitext(fname)[0]
                        })
                    docs.extend(loaded_docs)
                except Exception as e:
                    print(f"❌ Failed to process: {file_path}")
                    traceback.print_exc()

        if not docs:
            print(f"⚠️ No documents found in {folder_path}")
            continue

        print(f"📑 Loaded {len(docs)} documents from {subtype}")
        chunks = splitter.split_documents(docs)
        print(f"✂️ Split into {len(chunks)} chunks")

        db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

        batch, tokens, batch_id = [], 0, 1
        for doc in chunks:
            try:
                token_len = count_tokens(doc.page_content)
                if token_len > MAX_TOKENS_PER_CHUNK or token_len == 0:
                    continue
                if tokens + token_len > TOKEN_LIMIT:
                    db.add_documents([
                        Document(page_content=d.page_content, metadata=safe_filter_metadata(d.metadata))
                        for d in batch
                    ])
                    print(f"✅ {subtype}: Added batch {batch_id} ({len(batch)} docs)")
                    batch, tokens, batch_id = [], 0, batch_id + 1

                batch.append(doc)
                tokens += token_len
            except Exception as e:
                print(f"⚠️ Skipped one doc due to token error: {e}")

        if batch:
            db.add_documents([
                Document(page_content=d.page_content, metadata=safe_filter_metadata(d.metadata))
                for d in batch
            ])
            print(f"✅ {subtype}: Final batch {batch_id} added.")

        db.persist()
        print(f"💾 Chroma DB stored at: {chroma_path}")

if __name__ == "__main__":
    create_chroma_index()
