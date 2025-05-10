import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import os
import re
from typing import List, Dict, Any

# Custom embedding class to wrap SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Initialize components
embeddings = SentenceTransformerEmbeddings()
model = OllamaLLM(model="qwen2.5:3b", temperature=0.4)

# Path to Ramayana text file
RAMAYANA_FILE_PATH = "C:/PF/Projects/NYD/Datasets/Final_data.txt"
DB_PATH = "ramayana_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Enhanced fact-check template
fact_check_template = """
You are an expert on the Indian epic Ramayana. Your task is to verify if the user's statement is supported by the text of the Ramayana.

The user statement is: {statement}

Here are relevant excerpts from the Ramayana:
{context}

Based on the provided context from the Ramayana, determine if the user's statement is TRUE or FALSE.

You must carefully analyze the text to identify:
- The Book name (e.g., Bala Kanda, Ayodhya Kanda, etc.)
- The Sarga (chapter) number
- The Shloka (verse) number or range

Look for patterns like "[Book name like BALA,AYODHYA] KANDA, Sarga [Number], Shloka [Number]" or similar indicators in the context.

Your answer should follow this format:
VERDICT: [TRUE/FALSE]
[If TRUE] LOCATION: [Book Name], Sarga [Number], Shloka [Number/Range]
[If TRUE] EXPLANATION: [Brief explanation with direct quotation]
[If FALSE] EXPLANATION: [Why the statement is not supported by the text]

Answer:
"""
fact_check_prompt = ChatPromptTemplate.from_template(fact_check_template)

# Function to chunk text into manageable pieces with metadata
def chunk_text(text: str) -> List[Document]:
    # First, try to identify book, sarga, and shloka sections
    chunks = []
    
    # Simple chunking with overlap
    total_length = len(text)
    for i in range(0, total_length, CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_text = text[i:i + CHUNK_SIZE]
        
        # Extract metadata (book, sarga, shloka) if possible
        metadata = {}
        book_match = re.search(r'(Bala|Ayodhya|Aranya|Kishkindha|Sundara|Yuddha|Uttara)\s+Kanda', chunk_text)
        if book_match:
            metadata['book'] = book_match.group(0)
        
        sarga_match = re.search(r'Sarga\s+(\d+)', chunk_text)
        if sarga_match:
            metadata['sarga'] = sarga_match.group(1)
            
        shloka_match = re.search(r'Shlok\s+(\d+(?:-\d+)?)', chunk_text)
        if shloka_match:
            metadata['shlok'] = shloka_match.group(1)
            
        doc = Document(page_content=chunk_text, metadata=metadata)
        chunks.append(doc)
    
    return chunks

# Load or build FAISS DB with chunked documents
def load_db():
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    if not os.path.exists(RAMAYANA_FILE_PATH):
        st.error(f"File not found: {RAMAYANA_FILE_PATH}")
        return None
        
    with open(RAMAYANA_FILE_PATH, 'r', encoding='utf-8') as f:
        full_text = f.read()
        
    # Create chunked documents
    chunked_docs = chunk_text(full_text)
    
    # Create and save database
    db = FAISS.from_documents(chunked_docs, embeddings)
    db.save_local(DB_PATH)
    return db

# Enhanced verification function
def verify_statement(statement, db):
    # Get more relevant chunks for better context
    relevant_chunks = db.similarity_search(statement, k=3)
    
    # Combine context from relevant chunks
    combined_context = "\n\n".join([doc.page_content for doc in relevant_chunks])
    
    # Extract any metadata
    metadata_info = []
    for doc in relevant_chunks:
        if doc.metadata:
            meta_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
            metadata_info.append(f"[Metadata: {meta_str}]")
    
    # Add metadata hints if available
    if metadata_info:
        combined_context += "\n\nPossible reference information:\n" + "\n".join(metadata_info)
    
    # Run the verification
    chain = fact_check_prompt | model
    return chain.invoke({"statement": statement, "context": combined_context})

# Streamlit UI
def main():
    st.title("Ramayana Fact Checker")
    st.info(f"Using: {RAMAYANA_FILE_PATH}")

    if 'db' not in st.session_state:
        with st.spinner("Loading database... (This might take a few minutes on first run)"):
            st.session_state.db = load_db()
    if not st.session_state.db:
        return

    user_statement = st.text_area("Enter a statement about the Ramayana:", height=100)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        verify_button = st.button("Verify Statement")
    with col2:
        rebuild_db = st.button("Rebuild Database", help="Use this if you've updated the source text file")
    
    if rebuild_db:
        with st.spinner("Rebuilding database..."):
            if os.path.exists(DB_PATH):
                import shutil
                shutil.rmtree(DB_PATH)
            st.session_state.db = load_db()
        st.success("Database rebuilt successfully!")
        
    if verify_button:
        if not user_statement:
            st.warning("Please enter a statement to verify.")
        else:
            with st.spinner("Verifying your statement..."):
                result = verify_statement(user_statement, st.session_state.db)
            
            st.markdown("### Result")
            
            # Color-coded result
            if "VERDICT: TRUE" in result.upper():
                st.success("The statement is TRUE according to the Ramayana")
            else:
                st.error("The statement is FALSE or not verifiable in the Ramayana")
            
            # Display formatted result
            formatted_result = result.replace("VERDICT:", "**VERDICT:**").replace("LOCATION:", "**LOCATION:**").replace("EXPLANATION:", "**EXPLANATION:**")
            st.markdown(formatted_result)
            
            # Show debugging info in expander
            with st.expander("Show debugging information"):
                st.write("Statement analyzed:", user_statement)
                relevant_chunks = st.session_state.db.similarity_search(user_statement, k=3)
                for i, chunk in enumerate(relevant_chunks):
                    st.markdown(f"#### Relevant Text {i+1}")
                    st.text(chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content)
                    if chunk.metadata:
                        st.write("Metadata:", chunk.metadata)

if __name__ == "__main__":
    main()