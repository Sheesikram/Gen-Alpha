from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4",temperature=0.9,max_completion_tokens=100)

result = model.invoke("make a poem on  psycho? 1960")

print(result.content)


