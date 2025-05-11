import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
model = OllamaLLM(model="qwen2.5:7b", temperature=0.2)

# Path to Ramayana text file
RAMAYANA_FILE_PATH = "C:/PF/Projects/NYD/Datasets/Final_data.txt"
DB_PATH = "ramayana_db"

# Simple fact-check template
fact_check_template = """
You are an expert on the Indian epic Ramayana. Your task is to verify if the user's statement is supported by the text of the Ramayana.

The user statement is: {statement}
In the statement focus on the important keyword and not everything
you have to verify the important question in the statement and not the whole statement

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

Look where the statement was fist mentioned and give the first reference of the statement in the Ramayana
Only give one sarga and Shlok number if you find multiple references. 
dont summarise the context , give what is given in the Shlok . 
"""
fact_check_prompt = ChatPromptTemplate.from_template(fact_check_template)

# Load or build FAISS DB with simple chunking
def load_db():
    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
    
    if not os.path.exists(RAMAYANA_FILE_PATH):
        st.error(f"File not found: {RAMAYANA_FILE_PATH}")
        return None
        
    with open(RAMAYANA_FILE_PATH, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Use simple text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(full_text)
    
    # Create documents from chunks with minimal metadata
    docs = [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]
    
    # Create and save database
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_PATH)
    
    return db

# Simple verification function
def verify_statement(statement, db):
    # Get most relevant passages
    relevant_docs = db.similarity_search(statement, k=10,similarity_threshold=0.7)
    
    # Extract content from docs
    passages = [doc.page_content for doc in relevant_docs]
    
    # Combine passages
    combined_context = "\n\n---\n\n".join(passages)
    
    # Reasonable size limit for LLM
    MAX_CONTEXT_SIZE = 40000
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

    # Initialize database
    if 'db' not in st.session_state:
        with st.spinner("Loading database... (This might take a few minutes on first run)"):
            st.session_state.db = load_db()
    
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
            formatted_result = result.replace("VERDICT:", "**VERDICT:**").replace("EXPLANATION:", "**EXPLANATION:**")
            st.markdown(formatted_result)
            
            # Show debugging info in expander
            with st.expander("Show debugging information"):
                st.write("Statement analyzed:", user_statement)
                relevant_docs = st.session_state.db.similarity_search(user_statement, k=2)
                for i, doc in enumerate(relevant_docs):
                    st.markdown(f"#### Relevant Document {i+1}")
                    
                    if doc.metadata:
                        st.write("Metadata:", doc.metadata)
                    
                    # Show document preview
                    preview_length = min(1000, len(doc.page_content))
                    st.markdown(f"**Text Preview** (first {preview_length} characters)")
                    st.text(doc.page_content[:preview_length] + ("..." if len(doc.page_content) > preview_length else ""))

if __name__ == "__main__":
    main()