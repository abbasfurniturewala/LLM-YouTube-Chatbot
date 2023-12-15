
import openai
import re
import snowflake.connector
import langchain.chains
import streamlit as st
import textwrap

# from langchain.chains import LLMChain
# from langchain.chat_models import ChatOpenAI  
# from langchain.utilities import SQLDatabase 
# from langchain.chains import create_sql_query_chain
from snowflake.snowpark import Session


import openai
import re
# import streamlit as st
from prompts import get_system_prompt
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI  
from langchain.utilities import SQLDatabase 
from langchain.chains import create_sql_query_chain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


st.title("📺YOUTUBE BOT")

# Initialize the chat messages history
openai.api_key = st.secrets.OPENAI_API_KEY
if "messages" not in st.session_state:
    # system prompt includes table information, rules, and prompts the LLM to produce
    # a welcome message to the user.
    st.session_state.messages = [{"role": "system", "content": get_system_prompt()}]

# Prompt for user input and save
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

# display the existing chat messages
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "results" in message:
            st.dataframe(message["results"])


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=40)
        prompt_text = get_system_prompt()
        all_splits = text_splitter.split_text(prompt_text)
        vectorstore = FAISS.from_texts(texts=all_splits, embedding=OpenAIEmbeddings(openai_api_key=st.secrets.OPENAI_API_KEY))

        prompt = ChatPromptTemplate(messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),]
)

        # Use the RAG components for the chat
        gpt = ChatOpenAI(temperature=0, openai_api_key=st.secrets.OPENAI_API_KEY, model_name='gpt-4-1106-preview')
        memory = ConversationBufferMemory(llm=gpt, memory_key="chat_history", return_messages=True)
        retriever = vectorstore.as_retriever()
        question_generator_chain= LLMChain(llm=gpt, prompt=prompt)
        chain = ConversationalRetrievalChain(gpt, retriever=retriever, memory=memory, prompt = prompt)

        # Process the input using RAG
        response = chain.run([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])

        # Display and process the response
        resp_container = st.empty()
        resp_container.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


















# text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=40)
# prompt_text = get_system_prompt()

# all_splits = text_splitter.split_text(prompt_text)

# vectorstore = FAISS.from_texts(texts=all_splits, embedding=OpenAIEmbeddings(openai_api_key='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'))

# prompt = ChatPromptTemplate(messages=[
#         SystemMessagePromptTemplate.from_template(
#             "You are a nice chatbot having a conversation with a human."
#         ),
#         # The `variable_name` here is what must align with memory
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{question}"),]
# )
# #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# gpt = ChatOpenAI(temperature=0, openai_api_key="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", model_name= 'gpt-4-1106-preview')

# memory = ConversationSummaryMemory(
#     llm=gpt, memory_key="chat_history", return_messages=True
# )

# retriever = vectorstore.as_retriever()
# qa = ConversationalRetrievalChain.from_llm(gpt, retriever=retriever, memory=memory)

# answer = qa("what does the app do ?")
# answers = answer["answer"]
# wrapped_text = textwrap.fill(answers, width=80)  # Set width as per requirement
# print(wrapped_text)

# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         # Initialize Langchain chain
#         # prompt = {"role": "system", "content": get_system_prompt()}

#         prompt = ChatPromptTemplate(messages=[
#         SystemMessagePromptTemplate.from_template(
#             "You are a nice chatbot having a conversation with a human."
#         ),
#         # The `variable_name` here is what must align with memory
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{question}"),]
# )
#         memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         gpt = ChatOpenAI(temperature=0, openai_api_key=st.secrets.OPENAI_API_KEY, model_name= 'gpt-4-1106-preview')
#         llm_chain = LLMChain(llm=gpt, prompt=prompt, verbose=True, memory=memory)  # adjust the model as necessary
#         llm_chain = ConversationalRetrievalChain.from_llm(gpt, retriever=retriever, memory=memory)
    
#         # Process the input using Langchain chain
#         response = llm_chain.run([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])

#         # Display the response
#         resp_container = st.empty()
#         resp_container.markdown(response)

#         message = {"role": "assistant", "content": response}
#         # Parse the response for a SQL query and execute if available
#         # ... existing SQL processing logic ...

#         # Parse the response for a SQL query and execute if available
#         sql_match = re.search(r"```sql\n(.*)\n```", response, re.DOTALL)
#         if sql_match:
#             sql = sql_match.group(1)
#             conn = st.experimental_connection("snowpark")
#             sql_results = conn.query(sql)
#             message["results"] = sql_results
#             st.dataframe(message["results"])

#             query_results_str = str(sql_results)

#             sql_results_message = {"role": "assistant", "content": f"SQL Query Results: {query_results_str}"}

#             # Append the SQL results message to the conversation history
#             st.session_state.messages.append(sql_results_message)
