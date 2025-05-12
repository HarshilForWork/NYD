import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Custom embedding class using multi-qa-MiniLM-L6-cos-v1 model
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="multi-qa-MiniLM-L6-cos-v1"):
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
DB_PATH = "ramayana_db_mini"  # Changed DB path to avoid conflicts with previous DB

# Improved fact-check template with more specific instructions
fact_check_template = """
You are an expert on the Indian epic Ramayana. Your task is to verify if the user's statement is supported by the text of the Ramayana.

The user statement is: {statement}

Focus on the key claims in the statement rather than minor details. Identify the most important facts that need verification.

Here are relevant excerpts from the Ramayana:
{context}

Based on the provided context from the Ramayana, determine if the user's statement is TRUE or FALSE.

If you find direct evidence supporting the statement, cite it verbatim.

Your answer should follow this format:
VERDICT: [TRUE/FALSE]
BOOK: [Book Name]
SARGA: [Sarga Number]
SHLOK: [Shlok Number]
EXPLANATION: [Brief explanation with direct quotation if applicable]

Important guidelines:
1. Cite only the first occurrence where the statement is mentioned in the Ramayana
2. Provide a single sarga and shlok number even if multiple references exist
3. Quote directly from the text rather than summarizing
4. Be precise about which book, sarga, and shlok contains the evidence
5. If the statement contains multiple claims, focus on verifying the main claim
"""
fact_check_prompt = ChatPromptTemplate.from_template(fact_check_template)

# Load or build FAISS DB with improved chunking
def load_db():
    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
    
    if not os.path.exists(RAMAYANA_FILE_PATH):
        st.error(f"File not found: {RAMAYANA_FILE_PATH}")
        return None
        
    with open(RAMAYANA_FILE_PATH, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Using normal text splitting with default parameters
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

# Improved verification function
def verify_statement(statement, db):
    # Get more relevant passages with better filtering
    relevant_docs = db.similarity_search(statement, k=12, fetch_k=20)
    
    # Extract content from docs
    passages = [doc.page_content for doc in relevant_docs]
    
    # Combine passages
    combined_context = "\n\n---\n\n".join(passages)
    
    # Reasonable size limit for LLM with smarter truncation
    MAX_CONTEXT_SIZE = 18000
    if len(combined_context) > MAX_CONTEXT_SIZE:
        # Keep first 3 documents in full as they're likely most relevant
        first_docs = "\n\n---\n\n".join(passages[:3])
        remaining_space = MAX_CONTEXT_SIZE - len(first_docs) - 100  # Buffer for joining text
        
        # Use remaining space for snippets of other documents
        if remaining_space > 0:
            num_remaining_docs = len(passages) - 3
            if num_remaining_docs > 0:
                chars_per_doc = remaining_space // num_remaining_docs
                remaining_docs = []
                for doc in passages[3:]:
                    if chars_per_doc < 200:  # If space is too small, skip
                        continue
                    remaining_docs.append(doc[:chars_per_doc] + "...")
                
                combined_context = first_docs + "\n\n---\n\n" + "\n\n---\n\n".join(remaining_docs)
            else:
                combined_context = first_docs
        else:
            combined_context = first_docs[:MAX_CONTEXT_SIZE]
    
    # Run the verification
    chain = fact_check_prompt | model
    return chain.invoke({"statement": statement, "context": combined_context})

# Streamlit UI with improved user experience
def main():
    st.set_page_config(page_title="Ramayana Fact Checker", page_icon="📜", layout="wide")
    
    st.title("📜 Ramayana Fact Checker")
    st.markdown("""
    This app verifies statements against the text of the Ramayana using AI.
    Enter your statement below and click "Verify" to check its accuracy.
    """)
    
    st.info(f"Source text: {RAMAYANA_FILE_PATH}")

    # Initialize database
    if 'db' not in st.session_state:
        with st.spinner("Loading database... (This might take a few minutes on first run)"):
            st.session_state.db = load_db()
    
    # Check if database loaded successfully
    if st.session_state.db is None:
        st.error("Failed to load database.")
        return

    # Example statements for user guidance
    with st.expander("See example statements to verify"):
        st.markdown("""
        - "Rama broke the bow of Lord Shiva during Sita's swayamvara."
        - "Hanuman set fire to Lanka using his tail."
        - "Kaikeyi had three boons from King Dasharatha."
        - "Lakshmana drew a protective line (Lakshmana Rekha) around Sita's hut."
        - "Ravana had twenty arms and ten heads."
        """)

    user_statement = st.text_area("Enter a statement about the Ramayana:", height=100,
                                 placeholder="Example: Rama was exiled to the forest for fourteen years.")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        verify_button = st.button("🔍 Verify Statement", use_container_width=True)
    with col2:
        rebuild_db = st.button("🔄 Rebuild Database", help="Use this if you've updated the source text file", use_container_width=True)
    
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
            with st.spinner("Verifying your statement... (This may take a moment)"):
                result = verify_statement(
                    user_statement, 
                    st.session_state.db
                )
            
            st.markdown("### Result")
            
            # Color-coded result with improved formatting
            if "VERDICT: TRUE" in result.upper():
                st.success("✅ The statement is TRUE according to the Ramayana")
            else:
                st.error("❌ The statement is FALSE or not verifiable in the Ramayana")
            
            # Display formatted result
            formatted_result = (result
                .replace("VERDICT:", "**VERDICT:**")
                .replace("BOOK:", "**BOOK:**")
                .replace("SARGA:", "**SARGA:**")
                .replace("SHLOK:", "**SHLOK:**")
                .replace("EXPLANATION:", "**EXPLANATION:**"))
            
            # Create a nice looking box for the result
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6;">
                {formatted_result}
            </div>
            """, unsafe_allow_html=True)
            
            # Show debugging info in expander
            with st.expander("Show technical details"):
                st.write("Statement analyzed:", user_statement)
                
                # Show top relevant passages
                st.markdown("#### Top Relevant Passages")
                relevant_docs = st.session_state.db.similarity_search(user_statement, k=3)
                
                for i, doc in enumerate(relevant_docs):
                    st.markdown(f"##### Passage {i+1}")
                    
                    if doc.metadata:
                        st.write("Metadata:", doc.metadata)
                    
                    # Show document preview in a scrollable container
                    st.markdown(f"""
                    <div style="max-height: 200px; overflow-y: auto; padding: 10px; 
                                border: 1px solid #ddd; border-radius: 5px; 
                                margin-bottom: 15px; background-color: #f9f9f9; font-family: monospace;">
                        {doc.page_content.replace('\n', '<br>')}
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()