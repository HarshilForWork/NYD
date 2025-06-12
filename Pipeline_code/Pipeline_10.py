"""
The application reads the full Ramayana text and extracts individual shlokas using regex, 
storing each as a separate document in a FAISS vector database along with metadata such as book name, sarga number, 
and shlok number to preserve context. These documents are embedded using a sentence-transformer model for semantic search. 
When a user uploads a CSV with statements, the app performs similarity search to retrieve the most relevant shloks for each statement.
For each result, approximately 1000 characters before and after the retrieved vector are also included to provide contextual clarity.
This combined context and the user's statement are passed to a large language model (LLM) via a structured prompt,
which returns a verdict (TRUE/FALSE) and saves results to a CSV file.
"""
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import os
import re
import subprocess
import time
from typing import List, Dict, Any, Optional
from io import StringIO

# Custom embedding class to wrap SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Function to check if Ollama model exists and pull if needed
def ensure_ollama_model(model_name="qwen2.5:14b"):
    """
    Check if the specified Ollama model exists, and pull it if it doesn't.
    
    Args:
        model_name: Name of the Ollama model to check/pull
    
    Returns:
        bool: True if model is available, False if failed to pull
    """
    try:
        # Check if model exists
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        if result.returncode == 0:
            # Check if our model is in the list
            if model_name in result.stdout:
                st.success(f"✅ Model '{model_name}' is already available")
                return True
            else:
                st.warning(f"⚠️ Model '{model_name}' not found. Attempting to pull...")
                
                # Pull the model
                with st.spinner(f"Pulling model '{model_name}'... This may take several minutes."):
                    pull_result = subprocess.run(
                        ["ollama", "pull", model_name], 
                        capture_output=True, 
                        text=True, 
                        timeout=18000  # 5 hours timeout for pulling large models
                    )
                
                if pull_result.returncode == 0:
                    st.success(f"✅ Successfully pulled model '{model_name}'")
                    return True
                else:
                    st.error(f"❌ Failed to pull model '{model_name}': {pull_result.stderr}")
                    return False
        else:
            st.error(f"❌ Failed to check Ollama models: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        st.error("❌ Timeout while checking/pulling Ollama model")
        return False
    except FileNotFoundError:
        st.error("❌ Ollama is not installed or not in PATH")
        return False
    except Exception as e:
        st.error(f"❌ Error checking Ollama model: {str(e)}")
        return False

# Initialize components
embeddings = SentenceTransformerEmbeddings()

# Path to Ramayana text file
RAMAYANA_FILE_PATH = "NYD/Final_data.txt"
DB_PATH = "ramayana_db"

# Improved fact-check template
fact_check_template = """
You are an expert on the Indian epic Ramayana. Your task is to verify if the user's statement is supported by the text of the Ramayana.

The user statement is: {statement}
Focus on the important keywords and central claim in the statement.

Here are relevant excerpts from the Ramayana:
{context}

Based on the provided context from the Ramayana, determine if the user's statement is TRUE or FALSE.

Instructions:
- If the statement is supported by the context, respond with: VERDICT: TRUE
- If the statement contradicts or is not supported by the context, respond with: VERDICT: FALSE
- You must choose either TRUE or FALSE - do not use any other words.
- Don't be very serious about the question, understand the context and the statement, and give a verdict based on the text.
- Understand the meaning of the statement, grab the feel of it and then give a verdict.


VERDICT:"""
fact_check_prompt = ChatPromptTemplate.from_template(fact_check_template)

class ShlokExtractor:
    """Class to extract individual shloks from Ramayana text."""
    
    def __init__(self):
        # Regex patterns for extracting metadata and shloks
        self.book_pattern = re.compile(r'(BALA|AYODHYA|ARANYA|KISHKINDHA|SUNDARA|YUDDHA)\s+KANDA')
        self.sarga_pattern = re.compile(r'SARGA\s+(\d+)')
        # Improved shlok pattern to handle complex numbering like "12- 13a", "13b- 14a", etc.
        self.shlok_pattern = re.compile(r'Shlok\s+([\dab\-\s]+):\s*(.*?)(?=Shlok\s+[\dab\-\s]+:|$)', re.DOTALL)

    def extract_shloks(self, text: str) -> List[Dict[str, Any]]:
        """Extract individual shloks with their metadata from the text."""
        results = []
        current_book = "Unknown"
        current_sarga = "Unknown"
        
        # Find all book sections
        book_sections = self.split_by_pattern(text, r'(BALA|AYODHYA|ARANYA|KISHKINDHA|SUNDARA|YUDDHA)\s+KANDA')
        
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
                results.extend(self.extract_shloks_from_sarga(
                    sarga_section, current_book, current_sarga
                ))
        
        return results
    
    def extract_shloks_from_sarga(self, sarga_text: str, book: str, sarga: str) -> List[Dict[str, Any]]:
        """Extract shloks from a single sarga."""
        shloks = []
        
        # Find all shlok matches
        shlok_matches = list(self.shlok_pattern.finditer(sarga_text))
        
        for i, match in enumerate(shlok_matches):
            shlok_num = match.group(1).strip()
            shlok_text = match.group(2).strip()
            
            # Clean up the shlok text
            shlok_text = re.sub(r'\s+', ' ', shlok_text)
            
            # Skip if shlok text is too short
            if len(shlok_text.strip()) < 10:
                continue
            
            shlok_data = {
                "book": book,
                "sarga": sarga,
                "shlok": shlok_num,
                "text": shlok_text,
                "context": f"Book: {book}, Sarga: {sarga}, Shlok: {shlok_num}"
            }
            
            shloks.append(shlok_data)
        
        return shloks
    
    def split_by_pattern(self, text: str, pattern: str) -> List[str]:
        """Split text by regex pattern while preserving the pattern in each chunk."""
        regex = re.compile(pattern)
        split_positions = [0] + [m.start() for m in regex.finditer(text)]
        
        if len(split_positions) <= 1:
            return [text]
            
        result = []
        for i in range(len(split_positions) - 1):
            result.append(text[split_positions[i]:split_positions[i+1]])
        
        result.append(text[split_positions[-1]:])
        return result

# Load or build FAISS DB
def load_db():
    if os.path.exists(DB_PATH):
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
    
    if not os.path.exists(RAMAYANA_FILE_PATH):
        st.error(f"File not found: {RAMAYANA_FILE_PATH}")
        return None
        
    with open(RAMAYANA_FILE_PATH, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Extract individual shloks
    extractor = ShlokExtractor()
    shloks = extractor.extract_shloks(full_text)
    
    st.info(f"Extracted {len(shloks)} individual shloks from the text")
    
    # Create documents
    docs = []
    for i, shlok in enumerate(shloks):
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
    
    # Create and save database
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_PATH)
    
    return db

# Verification function
def verify_statement(statement, db, model):
    try:
        # Get relevant passages
        relevant_docs = db.similarity_search(statement, k=5)
        
        # Extract content
        passages = [doc.page_content for doc in relevant_docs]
        combined_context = "\n\n---\n\n".join(passages)
        
        # Limit context size
        MAX_CONTEXT_SIZE = 4000
        if len(combined_context) > MAX_CONTEXT_SIZE:
            half_size = MAX_CONTEXT_SIZE // 2
            combined_context = combined_context[:half_size] + "\n\n[...content omitted...]\n\n" + combined_context[-half_size:]
        
        # Run verification
        chain = fact_check_prompt | model
        result = chain.invoke({"statement": statement, "context": combined_context})
        
        # Debug: Print the raw result for troubleshooting
        if st.session_state.get('debug_mode', False):
            st.text(f"Raw LLM Response: {result}")
        
        # More robust verdict extraction
        result_upper = result.upper().strip()
        
        # Try multiple patterns to extract verdict
        if "VERDICT: TRUE" in result_upper or "TRUE" in result_upper.split()[-3:]:
            return "TRUE"
        elif "VERDICT: FALSE" in result_upper or "FALSE" in result_upper.split()[-3:]:
            return "FALSE"
        elif "TRUE" in result_upper and "FALSE" not in result_upper:
            return "TRUE"
        elif "FALSE" in result_upper and "TRUE" not in result_upper:
            return "FALSE"
        else:
            # Log the problematic response for debugging
            st.warning(f"Could not parse verdict from: {result[:100]}...")
            return "UNKNOWN"
            
    except Exception as e:
        st.error(f"Error verifying statement: {str(e)}")
        return "ERROR"

# Process CSV function
def process_csv_statements(df, db, model):
    """Process statements from CSV and return results."""
    results = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, row in df.iterrows():
        statement = row['Statement']
        
        # Update progress
        progress = (i + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Processing statement {i + 1} of {len(df)}: {statement[:50]}...")
        
        # Verify statement
        prediction = verify_statement(statement, db, model)
        
        results.append({
            'Statement': statement,
            'Prediction': prediction
        })
        
        # Add small delay to prevent overwhelming the model
        time.sleep(0.1)
    
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    return pd.DataFrame(results)

# Streamlit UI
def main():
    st.title("Ramayana Fact Checker - CSV Batch Processing")
    st.info("Upload a CSV file with 'Statement' column to batch process fact-checking")
    
    # Model selection
    model_name = st.selectbox(
        "Select Ollama Model:",
        ["qwen2.5:14b"],
        index=0
    )
    
    # Check and ensure model availability
    if st.button("Check/Pull Model"):
        if ensure_ollama_model(model_name):
            st.session_state['model_ready'] = True
            st.session_state['model_name'] = model_name
        else:
            st.session_state['model_ready'] = False
    
    # Initialize model if ready
    if st.session_state.get('model_ready', False):
        try:
            model = OllamaLLM(model=st.session_state['model_name'], temperature=0.6)
            st.success(f"✅ Model '{st.session_state['model_name']}' is ready!")
        except Exception as e:
            st.error(f"❌ Error initializing model: {str(e)}")
            st.session_state['model_ready'] = False
    
    # Database loading
    if 'db' not in st.session_state:
        with st.spinner("Loading database..."):
            st.session_state.db = load_db()
    
    if st.session_state.db is None:
        st.error("Failed to load database.")
        return
    
    st.success("✅ Database loaded successfully!")
    
    # File upload section
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader(
        "Upload CSV with 'Statement' column", 
        type=['csv'],
        help="CSV file should have a column named 'Statement' with statements to verify"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate CSV format
            if 'Statement' not in df.columns:
                st.error("❌ CSV must have a 'Statement' column")
                return
            
            # Show preview
            st.subheader("CSV Preview")
            st.dataframe(df.head())
            st.info(f"Found {len(df)} statements to process")
            
            # Process button
            if st.button("Process All Statements", disabled=not st.session_state.get('model_ready', False)):
                if not st.session_state.get('model_ready', False):
                    st.error("❌ Please check/pull the model first!")
                    return
                
                with st.spinner("Processing statements..."):
                    results_df = process_csv_statements(df, st.session_state.db, model)
                
                # Display results
                st.subheader("Results")
                st.dataframe(results_df)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total", len(results_df))
                with col2:
                    st.metric("TRUE", len(results_df[results_df['Prediction'] == 'TRUE']))
                with col3:
                    st.metric("FALSE", len(results_df[results_df['Prediction'] == 'FALSE']))
                with col4:
                    st.metric("ERRORS", len(results_df[results_df['Prediction'].isin(['UNKNOWN', 'ERROR'])]))
                
                # Download button
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv_data,
                    file_name="ramayana_fact_check_results.csv",
                    mime="text/csv"
                )   
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # Single statement testing
    st.header("Test Single Statement")
    
    # Debug mode toggle
    debug_mode = st.checkbox("Enable Debug Mode (shows raw LLM responses)")
    st.session_state['debug_mode'] = debug_mode
    
    test_statement = st.text_area("Enter a statement to test:", height=100)
    
    if st.button("Test Statement", disabled=not st.session_state.get('model_ready', False)):
        if test_statement and st.session_state.get('model_ready', False):
            with st.spinner("Verifying statement..."):
                result = verify_statement(test_statement, st.session_state.db, model)
            
            if result == "TRUE":
                st.success(f"✅ VERDICT: {result}")
            elif result == "FALSE":
                st.error(f"❌ VERDICT: {result}")
            else:
                st.warning(f"⚠️ VERDICT: {result}")

if __name__ == "__main__":
    main()