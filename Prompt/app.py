import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
import os
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Cricket Guru",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI with black text
st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #f0f2f6;
        color: #000000;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        width: 100%;
        margin: 5px 0;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    
    /* Text color for all elements */
    .stMarkdown, .stText, .stChatMessageContent, .stChatMessage {
        color: #000000 !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #000000 !important;
    }
    
    /* Input text */
    .stTextInput>div>div>input {
        color: #000000 !important;
    }
    
    /* Subheader text */
    .stSubheader {
        color: #000000 !important;
    }
    
    /* Links */
    a {
        color: #000000 !important;
        text-decoration: underline;
    }
    
    /* Chat input placeholder */
    .stChatInputContainer input::placeholder {
        color: #666666 !important;
    }
    
    /* Sidebar text */
    .css-1d391kg {
        color: #000000 !important;
    }
    
    /* Make sure all text in the app is black */
    * {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat history, selected question, and API key
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# Initialize the model with temperature for more dynamic responses
@st.cache_resource
def get_model():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("⚠️ GROQ API key not found in environment variables")
        return None
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.7
    )

# Create chat prompt template with better system prompt
@st.cache_resource
def get_chat_prompt():
    return ChatPromptTemplate([
        ("system", """You are an expert cricket knowledge assistant. Your role is to:
        1. Provide accurate and engaging information about cricket
        2. Share interesting facts and statistics
        3. Explain cricket concepts in simple terms
        4. Be enthusiastic and passionate about cricket
        5. Maintain a friendly and conversational tone
        Always be informative and entertaining while staying accurate."""),
    MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])

# Function to save chat history
def save_chat_history():
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.messages, f)

# Function to load chat history
def load_chat_history():
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r") as f:
            return json.load(f)
    return []

# Function to process query
def process_query(query):
    # Check if API key is set
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ API key not found in environment variables")
        return

    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
        st.caption(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Get model response with typing animation
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            # Get model and prompt template
            model = get_model()
            if model is None:
                return
                
            chat_prompt = get_chat_prompt()
            
            # Create prompt with chat history
            prompt = chat_prompt.invoke({
                "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages[:-1]],
                "query": query
            })
            
            # Get response
            response = model.invoke(prompt)
            
            # Display response with typing animation
            message_placeholder = st.empty()
            full_response = response.content
            
            # Simulate typing effect
            for i in range(len(full_response) + 1):
                message_placeholder.markdown(full_response[:i])
                time.sleep(0.02)
            
            st.caption(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.content,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Save updated chat history
            save_chat_history()

# Load previous chat history
if not st.session_state.messages:
    st.session_state.messages = load_chat_history()

# Title and description with emojis
st.title("🏏 CricGPT (Cricket Expert)")

st.markdown("""
    👋 Welcome to your CricGPT a Cricket Knowledge Assistant! 
    
    🎯 Ask me anything about:
    - Cricket players and their achievements
    - Match statistics and records
    - Cricket rules and formats
    - Historical matches and tournaments
    - Cricket terminology and concepts
    
    💡 I'll provide you with accurate, engaging, and interesting information!
""")

# Create two columns for layout
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat messages with animations
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"🕒 {message['timestamp']}")

    # Chat input with placeholder
    if query := st.chat_input("Ask me anything about cricket...", key="chat_input"):
        process_query(query)

    # Handle selected sample question
    if st.session_state.selected_question:
        process_query(st.session_state.selected_question)
        st.session_state.selected_question = None

with col2:
    # Sidebar features with better styling
    st.subheader("🎮 Chat Features")
    
    # Clear chat button with confirmation
    if st.button("🗑️ Clear Chat History"):
        if st.warning("Are you sure you want to clear the chat history?"):
            st.session_state.messages = []
            save_chat_history()
            st.rerun()
    
    # Download chat history
    if st.session_state.messages:
        chat_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}\nTime: {msg['timestamp']}"
            for msg in st.session_state.messages
        ])
        st.download_button(
            label="📥 Download Chat History",
            data=chat_text,
            file_name=f"cricket_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # Sample questions with categories
    st.subheader("💡 Sample Questions")
    
    categories = {
        "Players": [
            "Who is Virat Kohli?",
            "Tell me about Steve Smith's batting style",
            "What are Babar Azam's achievements?"
        ],
        "Tournaments": [
            "Who won the 2023 Cricket World Cup?",
            "Tell me about the IPL",
            "What is the Ashes series?"
        ],
        "Rules & Formats": [
            "Explain the rules of cricket",
            "What are the different formats of cricket?",
            "How does DLS method work?"
        ]
    }
    
    for category, questions in categories.items():
        st.markdown(f"**{category}**")
        for question in questions:
            if st.button(question, key=f"btn_{question}"):
                st.session_state.selected_question = question
                st.rerun()

# Footer with social links
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ by Shees Ikram With Langchain</p>
        <p>Follow me on: 
            <a href='https://github.com/Sheesikram'>GitHub</a> | 
            <a href='https://www.linkedin.com/in/shees-ikram/'>LinkedIn</a>
        </p>
    </div>
""", unsafe_allow_html=True)
