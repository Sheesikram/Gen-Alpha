from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
load_dotenv()
model=ChatOpenAI()
prompt1=PromptTemplate(
    template="Give me 5 Records of this cricketer name is {name}",
    input_variables=["name"]
)
model1= ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.7
    )

prompt2=PromptTemplate(
    template="Give me best Record from this {text}",
    input_variables=["text"]
)
perser=StrOutputParser()
chain=prompt1|model | perser | prompt2 | model1 | perser
result=chain.invoke({"name":"virat kohli"})
print(result)
chain.get_graph().print_ascii()