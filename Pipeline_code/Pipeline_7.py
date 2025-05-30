import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import os
import re
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import FieldCondition, MatchText, Filter
from qdrant_client.models import VectorParams, Distance, OptimizersConfigDiff, SearchParams

# New imports for NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

# Download necessary NLTK resources (only done once)
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

# Download NLTK resources
download_nltk_resources()

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

# Qdrant Cloud configuration
QDRANT_URL = "https://1a7850b1-fa20-43cc-b4cf-8568090b994c.eu-central-1-0.aws.cloud.qdrant.io:6333"  # Add your Qdrant Cloud URL here
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.WFZvs_Xj1QTBrkzktVYz4MCl1nJOBmB13YWhj_MywbA"  # Add your Qdrant Cloud API key here
USE_QDRANT_CLOUD = True  # Set to True to use Qdrant Cloud instead of local storage

# Path to Ramayana text file
RAMAYANA_FILE_PATH = "C:/PF/Projects/NYD/Datasets/Final_data.txt"
COLLECTION_NAME = "ramayana_collection"
VECTOR_SIZE = 768  # For all-mpnet-base-v2 model
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

def initialize_qdrant_client():
    """Initialize Qdrant client with local storage or cloud"""
    if USE_QDRANT_CLOUD and QDRANT_URL and QDRANT_API_KEY:
        # Use Qdrant Cloud
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        return client
    else:
        # Use local storage as fallback
        client = QdrantClient(path="./qdrant_storage")
        return client

def create_collection_if_not_exists(client, collection_name=COLLECTION_NAME):
    """Create Qdrant collection if it doesn't exist"""
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if collection_name not in collection_names:
            # Create new collection with specified parameters
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                ),
                # Enable payload indexing for text fields for keyword search
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=0  # Index immediately
                )
            )
            
            # Add payload index for text field (separate method call)
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="text",
                    field_schema="text"
                )
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="book",
                    field_schema="keyword"
                )
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="sarga",
                    field_schema="keyword"
                )
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name="shlok",
                    field_schema="keyword"
                )
            except Exception as e:
                st.warning(f"Note: Could not create payload index: {e}. Search will still work but might be slower.")
            
            return True  # New collection created
        return False  # Collection already exists
    except Exception as e:
        st.error(f"Error creating collection: {e}")
        return False

# Load or build Qdrant DB with individual shlok storage
def load_db():
    client = initialize_qdrant_client()
    
    try:
        # Try to get collection info to see if it exists
        try:
            collection_info = client.get_collection(COLLECTION_NAME)
            collection_exists = True
        except Exception:
            collection_exists = False
            
        # If collection doesn't exist, create it
        if not collection_exists:
            create_collection_if_not_exists(client)
        else:
            # Check if collection has points
            if collection_info.points_count > 0:
                return client
    except Exception as e:
        st.warning(f"Error checking collection: {e}. Will attempt to create/rebuild.")
        collection_exists = False
    
    # If we reach here, we need to build/rebuild the collection
    if not os.path.exists(RAMAYANA_FILE_PATH):
        st.error(f"File not found: {RAMAYANA_FILE_PATH}")
        return None
        
    with open(RAMAYANA_FILE_PATH, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Extract individual shloks with metadata
    extractor = ShlokExtractor()
    shloks = extractor.extract_shloks(full_text)
    
    # Create point objects for Qdrant
    points = []
    
    for i, shlok in enumerate(shloks):
        # Create document text
        doc_text = f"""Book: {shlok['book']} KANDA
        Sarga: {shlok['sarga']}
        Shlok {shlok['shlok']}: {shlok['text']}"""
        
        # Get embedding
        embedding = embeddings.embed_query(doc_text)
        
        # Create point
        point = models.PointStruct(
            id=i,
            vector=embedding,
            payload={
                "book": shlok['book'],
                "sarga": shlok['sarga'],
                "shlok": shlok['shlok'],
                "doc_id": i,
                "text": doc_text
            }
        )
        
        points.append(point)
        
        # Batch upload to avoid memory issues (every 100 points)
        if len(points) >= 100:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            points = []
            
    # Upload any remaining points
    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
    
    return client

# New function to extract keywords using NLTK
def extract_keywords_nltk(text, top_n=10):
    """
    Extract important keywords from text using NLTK's NLP capabilities
    without lemmatization
    
    Args:
        text: Input text
        top_n: Number of top keywords to return
        
    Returns:
        List of important keywords
    """
    # Tokenize text
    tokens = word_tokenize(text.lower())
    
    # Get stopwords and add punctuation
    stop_words = set(stopwords.words('english'))
    stop_words.update(punctuation)
    stop_words.update(['the', 'and', 'that', 'for', 'with', 'this', 'from', 
                       'have', 'was', 'were', 'are', 'said', 'will', 'also',
                       'they', 'their', 'there', 'would', 'could', 'should'])
    
    # Filter out stopwords and short words (without lemmatization)
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            filtered_tokens.append(token)
    
    # Count word frequencies
    word_freq = {}
    for token in filtered_tokens:
        if token in word_freq:
            word_freq[token] += 1
        else:
            word_freq[token] = 1
    
    # Sort by frequency and return top N
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_keywords[:top_n]]

# Hybrid search implementation with NLTK keyword extraction
def hybrid_search(client, query, k=5):
    """
    Perform hybrid search (combining vector search with keyword search)
    
    Args:
        client: Qdrant client
        query: User query
        k: Number of results to return
        
    Returns:
        List of relevant documents
    """
    # Get query embedding
    query_embedding = embeddings.embed_query(query)
    
    # Extract keywords using NLTK
    significant_keywords = extract_keywords_nltk(query, top_n=5)
    
    # Create keyword filter condition (if we have significant keywords)
    keyword_filters = []
    for keyword in significant_keywords:
        keyword_filters.append(
            models.FieldCondition(
                key="text",
                match=models.MatchText(text=keyword)
            )
        )
    
    # Combine filters with OR logic
    if keyword_filters:
        keyword_filter = models.Filter(
            should=keyword_filters  # OR logic between keywords
        )
    else:
        keyword_filter = None
    
    # Use hybrid scoring for better results (blend of vector and keyword relevance)
    # Set weights for vector search (semantic) and keyword search
    try:
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=keyword_filter,  # Add keyword-based filtering
            limit=k * 2,  # Get more results than needed, we'll rerank
            with_payload=True,
            with_vectors=False,
            score_threshold=0.5  # Minimum similarity threshold
        )
    except Exception as e:
        # Fallback to simple vector search without filter if keyword search fails
        st.warning(f"Keyword search failed: {e}. Falling back to vector search only.")
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=k * 2,
            with_payload=True,
            with_vectors=False,
            score_threshold=0.5
        )
    
    # Re-rank results to blend vector similarity with keyword presence
    scored_results = []
    for result in search_result:
        payload = result.payload
        text = payload["text"].lower()
        
        # Calculate keyword score based on presence of keywords in the text
        keyword_score = 0
        if significant_keywords:
            # Count matches for each keyword
            keyword_matches = sum(1 for keyword in significant_keywords if keyword.lower() in text)
            keyword_score = keyword_matches / len(significant_keywords)
        
        # Blend vector and keyword scores (70% vector, 30% keyword when keywords exist)
        if significant_keywords:
            final_score = (0.7 * result.score) + (0.3 * keyword_score)
        else:
            final_score = result.score
        
        scored_results.append((result, final_score))
    
    # Sort by combined score and take top k
    scored_results.sort(key=lambda x: x[1], reverse=True)
    top_results = scored_results[:k]
    
    # Extract documents from re-ranked results
    docs = []
    for result, final_score in top_results:
        payload = result.payload
        
        # Highlight matched keywords in text
        highlighted_text = payload["text"]
        for keyword in significant_keywords:
            pattern = re.compile(f"\\b{re.escape(keyword)}\\b", re.IGNORECASE)
            highlighted_text = pattern.sub(f"**{keyword.upper()}**", highlighted_text)
        
        doc = Document(
            page_content=highlighted_text,  # Use highlighted text in the doc
            metadata={
                "book": payload["book"],
                "sarga": payload["sarga"],
                "shlok": payload["shlok"],
                "doc_id": payload["doc_id"],
                "vector_score": result.score,
                "keyword_score": keyword_score if significant_keywords else None,
                "final_score": final_score,
                "matched_keywords": [k for k in significant_keywords if k.lower() in payload["text"].lower()] if significant_keywords else []
            }
        )
        docs.append(doc)
    
    return docs

# Verification function using hybrid search
def verify_statement(statement, client):
    # Get most relevant passages using hybrid search
    relevant_docs = hybrid_search(client, statement, k=5)
    
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
    
    # Qdrant Cloud configuration
    with st.sidebar:
        st.subheader("Qdrant Cloud Configuration")
        global QDRANT_URL, QDRANT_API_KEY, USE_QDRANT_CLOUD
        
        # Default to whatever is already set
        qdrant_url = st.text_input("Qdrant Cloud URL:", value=QDRANT_URL)
        qdrant_api_key = st.text_input("Qdrant API Key:", value=QDRANT_API_KEY, type="password")
        use_cloud = st.checkbox("Use Qdrant Cloud", value=USE_QDRANT_CLOUD)
        
        # Update global variables if values changed
        if qdrant_url != QDRANT_URL or qdrant_api_key != QDRANT_API_KEY or use_cloud != USE_QDRANT_CLOUD:
            QDRANT_URL = qdrant_url
            QDRANT_API_KEY = qdrant_api_key
            USE_QDRANT_CLOUD = use_cloud
            # Clear session state to force database reinitialization
            if 'db' in st.session_state:
                del st.session_state.db
                st.success("Qdrant configuration updated! Database will be reinitialized.")
    
    # Status indicator for database
    db_status = st.empty()

    # Initialize database
    if 'db' not in st.session_state:
        with st.spinner("Loading database... (This might take a few minutes on first run)"):
            db_status.warning("Loading database... Please wait.")
            st.session_state.db = load_db()
            if st.session_state.db:
                connection_type = "Qdrant Cloud" if USE_QDRANT_CLOUD and QDRANT_URL else "Local Qdrant"
                db_status.success(f"Database loaded successfully using {connection_type}!")
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
        rebuild_db = st.button("Rebuild Database", help="Use this if you've updated the source text file or changed Qdrant configuration")
    
    if rebuild_db:
        with st.spinner("Rebuilding database..."):
            db_status.warning("Rebuilding database... This may take several minutes.")
            # Delete existing collection
            client = st.session_state.db
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception as e:
                st.warning(f"Error deleting collection: {e}")
            
            # Rebuild database
            st.session_state.db = load_db()
            connection_type = "Qdrant Cloud" if USE_QDRANT_CLOUD and QDRANT_URL else "Local Qdrant"
            db_status.success(f"Database rebuilt successfully using {connection_type}!")
        
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
                
                # Extract and display keywords using NLTK
                extracted_keywords = extract_keywords_nltk(user_statement)
                st.write("NLTK extracted keywords:", extracted_keywords)
                
                relevant_docs = hybrid_search(st.session_state.db, user_statement, k=3)
                for i, doc in enumerate(relevant_docs):
                    st.markdown(f"#### Relevant Document {i+1}")
                    
                    if doc.metadata:
                        st.write("Vector Score:", round(doc.metadata.get("vector_score", 0), 3))
                        st.write("Keyword Score:", round(doc.metadata.get("keyword_score", 0), 3) if doc.metadata.get("keyword_score") is not None else "N/A")
                        st.write("Final Score:", round(doc.metadata.get("final_score", 0), 3))
                        st.write("Matched Keywords:", doc.metadata.get("matched_keywords", []))
                        st.write("Other Metadata:", {k: v for k, v in doc.metadata.items() 
                                                  if k not in ["vector_score", "keyword_score", "final_score", "matched_keywords"]})
                    
                    # Show document content
                    st.markdown("**Full Text:**")
                    st.markdown(doc.page_content)  # Using markdown to show highlighted keywords

if __name__ == "__main__":
    main()