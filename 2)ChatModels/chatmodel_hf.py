from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Verify HuggingFace API key is loaded
if not os.getenv("HUGGINGFACE_API_KEY"):
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")

# Initialize the model with a more stable model
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95
)
model=ChatHuggingFace(llm=llm)
try:
    # The response will be a string, so we print it directly
    result = model.invoke("who is the writer of psycho? 1960")
    print(result)
except Exception as e:
    print(f"An error occurred: {str(e)}")



