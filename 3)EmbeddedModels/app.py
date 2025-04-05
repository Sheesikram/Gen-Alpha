import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Cricket Document Search",
    page_icon="🏏",
    layout="wide"
)

# Title and description
st.title("🏏 Cricket Document Search")
st.markdown("""
This application helps you find information about different cricketers using semantic search.
Simply enter your question about any cricketer, and we'll find the most relevant information from our database.
""")

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

# Define our documents
docs = [
    "Virat Kohli is an Indian cricketer and former captain of the Indian national team. He is considered one of the best batsmen in the world.",
    "Steve Smith is an Australian cricketer known for his unorthodox batting style and exceptional test match batting average.",
    "Kane Williamson is a New Zealand cricketer who is the current captain of the New Zealand national team in all formats.",
    "Ben Stokes is an English cricketer who is known for his aggressive batting style and ability to perform in high-pressure situations.",
    "Babar Azam is a Pakistani cricketer who is considered one of the best batsmen in modern cricket across all formats."
]

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # User input
    query = st.text_input("Enter your question about cricketers:", 
                         placeholder="e.g., Who is Virat Kohli?")
    
    if query:
        # Get embeddings
        embeddings = get_embeddings()
        vector = embeddings.embed_documents(docs)
        question = embeddings.embed_query(query)
        
        # Calculate similarity
        similarity_search = cosine_similarity([question], vector)[0]
        i, sim = sorted(list(enumerate(similarity_search)), key=lambda x:x[1])[-1]
        
        # Display results
        st.subheader("Search Results")
        st.markdown("---")
        
        # Show the most relevant document
        st.markdown("**Most Relevant Information:**")
        st.info(docs[i])
        
        # Show similarity score
        st.markdown(f"**Similarity Score:** {sim:.4f}")
        
        # Show all documents with their similarity scores
        st.markdown("**All Documents (sorted by relevance):**")
        for idx, (doc_idx, score) in enumerate(sorted(list(enumerate(similarity_search)), 
                                                    key=lambda x:x[1], 
                                                    reverse=True)):
            st.markdown(f"{idx + 1}. {docs[doc_idx]}")
            st.markdown(f"   Similarity: {score:.4f}")
            st.markdown("---")

with col2:
    # Display information about available cricketers
    st.subheader("Available Cricketers")
    st.markdown("Our database contains information about:")
    for doc in docs:
        # Extract cricketer name from the document
        name = doc.split(" is ")[0]
        st.markdown(f"- {name}")
    
    # Add some styling
    st.markdown("""
    <style>
    .stMarkdown {
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Built by  Shees Ikram ❤️ using Streamlit")
