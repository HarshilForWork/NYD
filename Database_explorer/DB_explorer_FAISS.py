import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Path to your FAISS database
DB_PATH = "C:/PF/Projects/ramayana_db"

# Custom embedding class to wrap SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def inspect_document(doc_id=None, book=None, sarga=None, shlok=None):
    """
    View a single document from the FAISS database.
    
    Args:
        doc_id: Document ID to retrieve (integer for normal docs, string for context docs)
        book: Book name (e.g., "BALA", "AYODHYA")
        sarga: Sarga number
        shlok: Shlok number
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return
    
    print(f"Loading database from {DB_PATH}...")
    embeddings = SentenceTransformerEmbeddings()
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Get all documents from the docstore
    docstore = db.docstore
    all_docs = list(docstore._dict.values())
    
    # Find the document based on criteria
    found_doc = None
    
    if doc_id is not None:
        # Try as integer first (for regular docs)
        try:
            doc_id_int = int(doc_id)
            for doc in all_docs:
                if doc.metadata.get("doc_id") == doc_id_int:
                    found_doc = doc
                    break
        except ValueError:
            # Try as string (for context docs like "context_1")
            for doc in all_docs:
                if str(doc.metadata.get("doc_id")) == doc_id:
                    found_doc = doc
                    break
    elif book and sarga and shlok:
        # Find by book, sarga, and shlok
        book = book.upper()
        for doc in all_docs:
            if (doc.metadata.get("book", "").upper() == book and
                str(doc.metadata.get("sarga")) == str(sarga) and
                str(doc.metadata.get("shlok")) == str(shlok)):
                found_doc = doc
                break
    
    # Display the document if found
    if found_doc:
        print("\n===== Document Found =====")
        print(f"Metadata: {found_doc.metadata}")
        print("\nContent:")
        print(found_doc.page_content)
        print("="*30)
    else:
        print("No document found matching your criteria.")
        
        # Show what's available
        print("\nAvailable document statistics:")
        books = set(doc.metadata.get("book", "Unknown") for doc in all_docs)
        sargas = set(doc.metadata.get("sarga", "Unknown") for doc in all_docs)
        
        print(f"Books: {sorted(books)}")
        print(f"Sargas: {sorted(sargas, key=lambda x: int(x) if x.isdigit() else 9999)}")
        print(f"Total documents: {len(all_docs)}")

# Example usages:
if __name__ == "__main__":
    # To use this script, uncomment one of these examples:
    
    # Example 1: View document by ID
    inspect_document(doc_id="20")
    
    # Example 2: View document by book, sarga, and shlok
    #inspect_document(book="BALA", sarga="66", shlok="6")
    
    # Example 3: View a context document
    # inspect_document(doc_id="context_0")