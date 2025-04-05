from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model = ChatOpenAI()
 
template1=PromptTemplate(
    template="write a detailed report on this {topic} ",
    input_variables=["topic"]
)
template2=PromptTemplate(
    template="Make a 5 line summary about this {text}",
    input_variables=["text"]

)
template3=PromptTemplate(
    template="Now give me a name of this {text1}",
    input_variables=["text1"]

)
perser=StrOutputParser()

chain=template1 | model | perser | template2 | model | perser |template3 | model | perser
result=chain.invoke({"topic":"Blackhole"})
print(result)