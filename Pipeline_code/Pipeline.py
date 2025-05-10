import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import os

# Custom embedding class to wrap SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

# Initialize components
embeddings = SentenceTransformerEmbeddings()
model = OllamaLLM(model="qwen2.5:7b", temperature=0.9)

# Define the filepath to your local Ramayana text file
RAMAYANA_FILE_PATH ="C:/PF/Projects/NYD/Datasets/Final_data.txt"  # Update this with your actual file path

# Modified template to let the LLM extract the source information
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

Even if this information is not explicitly labeled in a structured format, use your understanding of the text to extract or infer this information from context clues, narrative structure, and content. Dont be harsh on the user and if the sentence is little bit off then also its alright and let it be true 

Your answer should follow this format:
VERDICT: [TRUE/FALSE]
[If TRUE] LOCATION: [Book Name], Sarga [Number], Shloka [Number/Range]
[If TRUE] EXPLANATION: [Brief explanation with direct quotation]
[If FALSE] EXPLANATION: [Why the statement is not supported by the text]

Answer:
"""

fact_check_prompt = ChatPromptTemplate.from_template(fact_check_template)

def setup_ramayana_db():
    """Load and process the Ramayana text file into a vector database without relying on regex."""
    if os.path.exists("ramayana_db"):
        # Load existing database
        return FAISS.load_local("ramayana_db", embeddings,allow_dangerous_deserialization=True)
    
    # Check if the Ramayana text file exists
    if not os.path.exists(RAMAYANA_FILE_PATH):
        st.error(f"The Ramayana text file does not exist at the specified path: {RAMAYANA_FILE_PATH}")
        st.error("Please update the RAMAYANA_FILE_PATH variable in the code with the correct path to your Ramayana text file.")
        return None
    
    # Read the file content
    with open(RAMAYANA_FILE_PATH, 'r', encoding='utf-8') as file:
        full_text = file.read()
    
    # Split the text into logical chunks for vector embedding
    # Don't try to parse structure here - let the LLM handle that
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    # Create document chunks
    docs = text_splitter.create_documents([full_text])
    
    # Create and save the vector database
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("ramayana_db")
    return db

def verify_statement(statement, db):
    """Verify the user's statement against the Ramayana database."""
    # Retrieve relevant documents
    relevant_docs = db.similarity_search(statement, k=5)
    
    # Extract context from retrieved documents
    context = ""
    for i, doc in enumerate(relevant_docs):
        context += f"Excerpt {i+1}:\n{doc.page_content}\n\n"
    
    # Generate the response using the LLM
    chain = fact_check_prompt | model
    response = chain.invoke({"statement": statement, "context": context})
    return response

# Streamlit interface
st.title("Ramayana Fact Checker")
st.write("Enter a statement about the Ramayana, and I'll check if it's true or false based on the text.")

# Show file path configuration at the top
st.info(f"Using Ramayana text file from: {RAMAYANA_FILE_PATH}")
st.write("To change the file path, update the `RAMAYANA_FILE_PATH` variable in the code.")

# Configuration section for file path
with st.expander("Configure your Ramayana file path"):
    st.markdown("""
        ### How to set up the file path:
        
        1. Open the Python script in a text editor
        2. Find the line: `RAMAYANA_FILE_PATH = "path/to/your/ramayana_complete.txt"`
        3. Replace with the actual path to your Ramayana text file
        4. Save the file and restart the application
        
        **Note:** Make sure to use the correct path format for your operating system:
        - Windows example: `"C:/Users/YourName/Documents/ramayana.txt"` or `"C:\\Users\\YourName\\Documents\\ramayana.txt"`
        - Mac/Linux example: `"/home/username/documents/ramayana.txt"`
    """)

# Initialize or load the vector database
if 'db' not in st.session_state:
    with st.spinner("Processing Ramayana text..."):
        st.session_state.db = setup_ramayana_db()

if st.session_state.db is None:
    st.error("Could not access the Ramayana text file. Please check the file path.")
else:
    st.success("Ramayana database loaded successfully!")
    
    # User input for the statement to verify
    user_statement = st.text_area("Enter a statement about the Ramayana:", 
                                 height=100,
                                 placeholder="Example: Hanuman burned the city of Lanka with his tail.")
    
    if st.button("Verify Statement"):
        if user_statement:
            with st.spinner("Verifying your statement..."):
                result = verify_statement(user_statement, st.session_state.db)
                
                # Display the result in a formatted way
                st.markdown("### Result")
                
                # Let the UI display the structured response from the LLM
                # Extract verdict using basic string search since we're not relying on regex
                if "VERDICT: TRUE" in result.upper():
                    st.success("The statement is TRUE according to the Ramayana")
                    
                    # Split the response into lines to find location and explanation
                    lines = result.split('\n')
                    location = ""
                    explanation = ""
                    
                    capture_explanation = False
                    for line in lines:
                        if line.upper().startswith("LOCATION:"):
                            location = line.replace("LOCATION:", "").strip()
                            st.info(f"**Source:** {location}")
                        elif line.upper().startswith("EXPLANATION:"):
                            explanation = line.replace("EXPLANATION:", "").strip()
                            capture_explanation = True
                        elif capture_explanation and line.strip():
                            explanation += " " + line.strip()
                    
                    if explanation:
                        st.write(f"**Details:** {explanation}")
                
                else:
                    st.error("The statement is FALSE or not verifiable in the Ramayana")
                    
                    # Extract explanation for FALSE verdict
                    lines = result.split('\n')
                    explanation = ""
                    capture_explanation = False
                    
                    for line in lines:
                        if line.upper().startswith("EXPLANATION:"):
                            explanation = line.replace("EXPLANATION:", "").strip()
                            capture_explanation = True
                        elif capture_explanation and line.strip():
                            explanation += " " + line.strip()
                    
                    if explanation:
                        st.write(f"**Reason:** {explanation}")
                
                # Show the full response in an expandable section
                with st.expander("View full analysis"):
                    st.write(result)
        else:
            st.warning("Please enter a statement to verify.")

    # Add a sample statements section
    with st.expander("Sample statements to try"):
        st.markdown("""
        - Hanuman burned the city of Lanka with his tail.
        - Ravana had ten heads.
        - Rama broke Lord Shiva's bow to win Sita's hand in marriage.
        - Kaikeyi had three boons to ask from King Dasharatha.
        - Lakshmana cut off Surpanakha's nose.
        """)

    # Information about file format
    with st.expander("About the Ramayana text"):
        st.markdown("""
        Your Ramayana text file can be in any format containing the complete text of the epic. The LLM will analyze the content to:

        1. Determine if statements are true or false based on the text
        2. Identify the relevant book, chapter (sarga), and verse (shloka) information when available
        3. Provide explanations with supporting text from the Ramayana

        The application processes your text file into semantic chunks that preserve context, allowing the LLM to reason about the content and structure of the epic.
        """)