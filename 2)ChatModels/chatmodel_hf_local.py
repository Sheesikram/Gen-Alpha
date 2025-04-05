from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature= 0.9, 
        max_new_tokens=100
        )
)
model=ChatHuggingFace(llm=llm)
result=model.invoke("who is the writer of psycho? 1960")

print(result)
