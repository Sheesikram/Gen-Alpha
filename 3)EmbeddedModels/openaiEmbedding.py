from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=32,
    api_key=os.getenv("OPENAI_API_KEY")
)

text="Hello, world!"

embedding=embeddings.embed_query(text)

print(str(embedding))


