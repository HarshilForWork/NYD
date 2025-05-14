import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import argparse

# Path to your FAISS database
DB_PATH = "C:/PF/Projects/NYD/ramayana_db"

# Custom embedding class to wrap SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def explore_db(db_path=DB_PATH, doc_id=None, book=None, sarga=None, shlok=None, search_query=None, limit=10):
    """
    Explore documents in FAISS database by various criteria.
    
    Args:
        db_path: Path to the FAISS database
        doc_id: Specific document ID to retrieve
        book: Filter documents by book (e.g., "BALA", "AYODHYA")
        sarga: Filter documents by sarga number
        shlok: Filter documents by shlok number
        search_query: Semantic search using this query
        limit: Maximum number of documents to display
    """
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
        
    # Load the database
    print(f"Loading database from {db_path}...")
    embeddings = SentenceTransformerEmbeddings()
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    # Get all document IDs (this is a bit of a hack but works)
    docstore = db.docstore
    all_docs = list(docstore._dict.values())
    
    # Filter by various criteria
    filtered_docs = all_docs.copy()
    
    if doc_id is not None:
        try:
            # Convert to int if it's a regular doc_id
            doc_id_int = int(doc_id)
            filtered_docs = [doc for doc in filtered_docs if doc.metadata.get("doc_id") == doc_id_int]
        except ValueError:
            # If conversion fails, it might be a context doc_id like "context_1"
            filtered_docs = [doc for doc in filtered_docs if str(doc.metadata.get("doc_id")) == doc_id]
    
    if book:
        book = book.upper()
        filtered_docs = [doc for doc in filtered_docs if doc.metadata.get("book", "").upper() == book]
    
    if sarga:
        filtered_docs = [doc for doc in filtered_docs if str(doc.metadata.get("sarga")) == str(sarga)]
    
    if shlok:
        # Check both direct shlok and shlok_range
        filtered_docs = [
            doc for doc in filtered_docs if (
                str(doc.metadata.get("shlok")) == str(shlok) or
                (doc.metadata.get("shlok_range") and str(shlok) in doc.metadata.get("shlok_range"))
            )
        ]
    
    # If search query is provided, do semantic search instead
    if search_query:
        docs = db.similarity_search(search_query, k=limit)
        filtered_docs = docs
    
    # Limit the number of documents to display
    docs_to_display = filtered_docs[:limit]
    
    # Display documents
    if not docs_to_display:
        print("No documents found matching your criteria.")
    else:
        print(f"Found {len(filtered_docs)} documents, showing {len(docs_to_display)}:")
        for i, doc in enumerate(docs_to_display):
            print(f"\n===== Document {i+1} =====")
            print(f"Metadata: {doc.metadata}")
            print("\nContent:")
            print(doc.page_content)
            print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore documents in a FAISS database")
    parser.add_argument("--db_path", default=DB_PATH, help="Path to the FAISS database")
    parser.add_argument("--doc_id", help="Look up a specific document by ID")
    parser.add_argument("--book", help="Filter by book (e.g., BALA, AYODHYA)")
    parser.add_argument("--sarga", help="Filter by sarga number")
    parser.add_argument("--shlok", help="Filter by shlok number")
    parser.add_argument("--search", help="Semantic search using a query")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of documents to display")
    
    args = parser.parse_args()
    
    explore_db(
        db_path=args.db_path,
        doc_id=args.doc_id,
        book=args.book,
        sarga=args.sarga,
        shlok=args.shlok,
        search_query=args.search,
        limit=args.limit
    )