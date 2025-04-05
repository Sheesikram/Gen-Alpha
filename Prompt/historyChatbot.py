from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model=ChatOpenAI()

#now create a chatprompttemplate
chat_Prompt=ChatPromptTemplate([
    ("system","you are a assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{queary}")
]
)

chat_history = []

with open("chat_History.txt") as f:
    chat_history.extend(f.readlines())

print(chat_history)
queastion=input("Enter queary of cricket")
#building the prompt
prompt=chat_Prompt.invoke({"chat_history":chat_history,"queary":queastion})
result=model.invoke(prompt)
print(result.content)
