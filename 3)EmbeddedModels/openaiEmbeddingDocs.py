from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=32,
    api_key=os.getenv("OPENAI_API_KEY")
)

docs={"Lahore is capital of Pakistan",
      "Islamabad is capital of Pakistan",
      "Karachi is capital of Pakistan",
      "Peshawar is capital of Pakistan",
      "Quetta is capital of Pakistan",
      "Rawalpindi is capital of Pakistan",
      "Faisalabad is capital of Pakistan",
         }

embedding=embeddings.embed_documents(docs)

print(str(embedding))


