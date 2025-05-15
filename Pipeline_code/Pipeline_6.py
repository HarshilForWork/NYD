import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import os
import re
from typing import List, Dict, Any, Optional

# Custom embedding class to wrap SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Initialize components
embeddings = SentenceTransformerEmbeddings()
model = OllamaLLM(model="qwen2.5:7b", temperature=0.2)

# Path to Ramayana text file
RAMAYANA_FILE_PATH = "C:/PF/Projects/NYD/Datasets/Final_data.txt"
DB_PATH = "ramayana_db"

# Improved fact-check template
fact_check_template = """
You are an expert on the Indian epic Ramayana. Your task is to verify if the user's statement is supported by the text of the Ramayana.

The user statement is: {statement}
Focus on the important keywords and central claim in the statement.

Here are relevant excerpts from the Ramayana:
{context}

Based on the provided context from the Ramayana, determine if the user's statement is TRUE or FALSE.

If you find direct evidence supporting the statement, cite it verbatim.

Your answer should follow this format:
VERDICT: [TRUE/FALSE]
BOOK: [Book Name]
Sarga: [Sarga Number]
Shlok: [Shlok Number]
EXPLANATION: [Brief explanation with direct quotation if applicable]

Give the first reference of the statement in the Ramayana.
Only provide one sarga and shlok number even if you find multiple references.
Don't summarize the context; quote directly what is given in the shlok.
"""
fact_check_prompt = ChatPromptTemplate.from_template(fact_check_template)

class ShlokExtractor:
    """Class to extract individual shloks from Ramayana text."""
    
    def __init__(self):
        # Regex patterns for extracting metadata and shloks
        self.book_pattern = re.compile(r'(BALA|AYODHYA|ARANYA|KISHKINDA|SUNDARA|YUDDHA)\s+KANDA')
        self.sarga_pattern = re.compile(r'SARGA\s+(\d+)')
        self.shlok_pattern = re.compile(r'Shlok\s+([\dab\-]+):\s*(.*?)(?=Shlok\s+[\dab\-]+:|$)', re.DOTALL)

    def extract_shloks(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract individual shloks with their metadata from the text.
        
        Args:
            text: The full Ramayana text
            
        Returns:
            List of dictionaries containing shlok text and metadata
        """
        results = []
        current_book = "Unknown"
        current_sarga = "Unknown"
        
        # Find all book sections
        book_sections = self.split_by_pattern(text, r'(BALA|AYODHYA|ARANYA|KISHKINDA|SUNDARA|YUDDHA)\s+KANDA')
        
        for book_section in book_sections:
            if not book_section.strip():
                continue
                
            # Extract book name
            book_match = self.book_pattern.search(book_section[:1000])
            if book_match:
                current_book = book_match.group(1)
            
            # Split book into sargas
            sarga_sections = self.split_by_pattern(book_section, r'SARGA\s+\d+')
            
            for sarga_section in sarga_sections:
                if not sarga_section.strip():
                    continue
                    
                # Extract sarga number
                sarga_match = self.sarga_pattern.search(sarga_section[:500])
                if sarga_match:
                    current_sarga = sarga_match.group(1)
                
                # Extract all shloks in this sarga
                shlok_matches = self.shlok_pattern.finditer(sarga_section)
                
                for match in shlok_matches:
                    shlok_num = match.group(1)
                    shlok_text = match.group(2).strip()
                    
                    # Store shlok with metadata
                    results.append({
                        "book": current_book,
                        "sarga": current_sarga,
                        "shlok": shlok_num,
                        "text": shlok_text,
                        # Include some context by adding nearby shlok numbers
                        "context": f"Book: {current_book}, Sarga: {current_sarga}, Shlok: {shlok_num}"
                    })
        
        return results
    
    def split_by_pattern(self, text: str, pattern: str) -> List[str]:
        """Split text by regex pattern while preserving the pattern in each chunk."""
        regex = re.compile(pattern)
        split_positions = [0] + [m.start() for m in regex.finditer(text)]
        
        if len(split_positions) <= 1:
            return [text]
            
        result = []
        for i in range(len(split_positions) - 1):
            result.append(text[split_positions[i]:split_positions[i+1]])
        
        # Add the last section
        result.append(text[split_positions[-1]:])
        
        return result

# Load or build FAISS DB with individual shlok storage
def load_db():
    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
    
    if not os.path.exists(RAMAYANA_FILE_PATH):
        st.error(f"File not found: {RAMAYANA_FILE_PATH}")
        return None
        
    with open(RAMAYANA_FILE_PATH, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Extract individual shloks with metadata
    extractor = ShlokExtractor()
    shloks = extractor.extract_shloks(full_text)
    
    # Create documents from individual shloks
    docs = []
    for i, shlok in enumerate(shloks):
        # For each shlok, create a document with its text and metadata
        doc_text = f"""Book: {shlok['book']} KANDA
        Sarga: {shlok['sarga']}
        Shlok {shlok['shlok']}: {shlok['text']}"""
        
        metadata = {
            "book": shlok['book'],
            "sarga": shlok['sarga'],
            "shlok": shlok['shlok'],
            "doc_id": i
        }
        
        docs.append(Document(page_content=doc_text, metadata=metadata))
    
    # Create and save database with individual shloks only
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_PATH)
    
    return db

# Simple verification function with improved context retrieval
def verify_statement(statement, db):
    # Get most relevant passages - increased k for better context
    relevant_docs = db.similarity_search(statement, k=5)  # Increased k to get more individual shloks for context
    
    # Extract content from docs
    passages = [doc.page_content for doc in relevant_docs]
    
    # Combine passages
    combined_context = "\n\n---\n\n".join(passages)
    
    # Reasonable size limit for LLM
    MAX_CONTEXT_SIZE = 1000
    if len(combined_context) > MAX_CONTEXT_SIZE:
        half_size = MAX_CONTEXT_SIZE // 2
        combined_context = combined_context[:half_size] + "\n\n[...content omitted for brevity...]\n\n" + combined_context[-half_size:]
    
    # Run the verification
    chain = fact_check_prompt | model
    return chain.invoke({"statement": statement, "context": combined_context})

# Streamlit UI
def main():
    st.title("Ramayana Fact Checker")
    st.info(f"Using: {RAMAYANA_FILE_PATH}")
    
    # Status indicator for database
    db_status = st.empty()

    # Initialize database
    if 'db' not in st.session_state:
        with st.spinner("Loading database... (This might take a few minutes on first run)"):
            db_status.warning("Loading database... Please wait.")
            st.session_state.db = load_db()
            if st.session_state.db:
                db_status.success("Database loaded successfully!")
            else:
                db_status.error("Failed to load database.")
    
    # Check if database loaded successfully
    if st.session_state.db is None:
        st.error("Failed to load database.")
        return

    user_statement = st.text_area("Enter a statement about the Ramayana:", height=100)

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        verify_button = st.button("Verify Statement")
    with col2:
        rebuild_db = st.button("Rebuild Database", help="Use this if you've updated the source text file")
    
    if rebuild_db:
        with st.spinner("Rebuilding database..."):
            db_status.warning("Rebuilding database... This may take several minutes.")
            if os.path.exists(DB_PATH):
                import shutil
                shutil.rmtree(DB_PATH)
            st.session_state.db = load_db()
            db_status.success("Database rebuilt successfully!")
        
    if verify_button:
        if not user_statement:
            st.warning("Please enter a statement to verify.")
        else:
            with st.spinner("Verifying your statement..."):
                result = verify_statement(
                    user_statement, 
                    st.session_state.db
                )
            
            st.markdown("### Result")
            
            # Color-coded result
            if "VERDICT: TRUE" in result.upper():
                st.success("The statement is TRUE according to the Ramayana")
            else:
                st.error("The statement is FALSE or not verifiable in the Ramayana")
            
            # Display formatted result
            formatted_result = result.replace("VERDICT:", "**VERDICT:**").replace("EXPLANATION:", "**EXPLANATION:**").replace("BOOK:", "**BOOK:**").replace("Sarga:", "**Sarga:**").replace("Shlok:", "**Shlok:**")
            st.markdown(formatted_result)
            
            # Show debugging info in expander
            with st.expander("Show debugging information"):
                st.write("Statement analyzed:", user_statement)
                relevant_docs = st.session_state.db.similarity_search(user_statement, k=3)
                for i, doc in enumerate(relevant_docs):
                    st.markdown(f"#### Relevant Document {i+1}")
                    
                    if doc.metadata:
                        st.write("Metadata:", doc.metadata)
                    
                    # Show document content
                    st.markdown("**Full Text:**")
                    st.text(doc.page_content)

if __name__ == "__main__":
    main()