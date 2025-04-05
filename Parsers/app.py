import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from typing import List, Dict
import json

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="DiReCT - Clinical Notes RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stChatMessage {
        background-color: black;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize the model
@st.cache_resource
def get_model():
    return ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

# Create chat prompt template
@st.cache_resource
def get_chat_prompt():
    return ChatPromptTemplate.from_template("""
    You are a medical expert assistant. Your role is to:
    1. Provide accurate and detailed information about medical conditions
    2. Explain medical concepts in simple terms
    3. Be cautious and suggest consulting healthcare professionals
    4. Maintain a professional and empathetic tone

    Context: {context}
    Question: {question}
    Answer: 
    """)

# Function to process clinical notes
def process_notes(notes: List[str]) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text("\n".join(notes))

# Function to create vector store
def create_vector_store(texts: List[str], embeddings):
    return FAISS.from_texts(texts, embeddings)

# Function to process query
def process_query(query: str, vector_store):
    # Get model and prompt template
    model = get_model()
    prompt = get_chat_prompt()
    
    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    
    # Get response
    response = qa_chain.invoke({"query": query})
    return response["result"]

# Main application
def main():
    st.title("🏥 DiReCT - Clinical Notes RAG System")
    
    # Sidebar for data upload
    with st.sidebar:
        st.subheader("📂 Upload Clinical Notes")
        uploaded_file = st.file_uploader("Upload your clinical notes (CSV or JSON)", type=["csv", "json"])
        
        if uploaded_file:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            
            # Process notes and create vector store
            if "notes" in df.columns:
                texts = process_notes(df["notes"].tolist())
                embeddings = get_embeddings()
                st.session_state.vector_store = create_vector_store(texts, embeddings)
                st.success("✅ Notes processed and indexed successfully!")
    
    # Main chat interface
    st.subheader("💬 Ask Questions About Clinical Notes")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if query := st.chat_input("Ask a question about the clinical notes..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process query and display response
        if st.session_state.vector_store:
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing..."):
                    response = process_query(query, st.session_state.vector_store)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning("⚠️ Please upload clinical notes first")

if __name__ == "__main__":
    main()