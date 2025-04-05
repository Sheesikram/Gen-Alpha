from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()
model1= ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.7
    )
parser = StrOutputParser()

class Review(BaseModel):
    sentiment:Literal["Pos","Neg"]=Field(default="pos",description="Give me the sentiment of this review")


perser_pydantic=PydanticOutputParser(pydantic_object=Review)




prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':perser_pydantic.get_format_instructions()}
)

classifier_chain = prompt1 | model | perser_pydantic

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Pos', prompt2 | model1 | parser),
    (lambda x:x.sentiment == 'Neg', prompt3 | model1 | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a worse phone'}))

chain.get_graph().print_ascii()