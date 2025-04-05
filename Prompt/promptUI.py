from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

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

st.header('Shees Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template=load_prompt("template.json")

result=template.invoke(
    {
        "paper_input":paper_input,
        "style_input":style_input,
        "length_input":length_input

    }
)


if st.button('Summarize'):
    r2=model.invoke(result)
    # chain = template | model
    # result = chain.invoke({
    #     'paper_input':paper_input,
    #     'style_input':style_input,
    #     'length_input':length_input
    # })
    st.write(r2.content)