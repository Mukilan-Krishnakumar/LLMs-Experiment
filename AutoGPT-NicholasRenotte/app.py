# Dependencies
import os

from langchain import chains
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

os.environ["OPENAI_API_KEY"] = apikey

# App framework
st.title("🦜 YouTube GPT Creator")
prompt = st.text_input("Insert the prompt")

# Prompt Templates
title_template = PromptTemplate(
    input_variables = ["topic"],
    template = "Write me a YouTube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables = ["title"],
    template = "Write me a YouTube video script based on this title TITLE: {title}"
)

# LLMS
llm = OpenAI(temperature = 0.9)
title_chain = LLMChain(llm=llm, prompt = title_template, verbose = True)
script_chain = LLMChain(llm = llm, prompt = script_template, verbose = True)
sequential_chain = SimpleSequentialChain(chains = [title_chain, script_chain], verbose = True)

# Output text
if prompt:
    response = sequential_chain.run(prompt)
    st.write(response)