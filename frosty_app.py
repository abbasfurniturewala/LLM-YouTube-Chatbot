import openai
import re
import streamlit as st
from prompts import get_system_prompt
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI  
from langchain.utilities import SQLDatabase 
from langchain.chains import create_sql_query_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


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

#If last message is not from assistant, we need to generate a new response
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         response = ""
#         resp_container = st.empty()
#         for delta in openai.ChatCompletion.create(
#             model="gpt-4-1106-preview",
#             messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
#             stream=True,
#         ):
#             response += delta.choices[0].delta.get("content", "")
#             resp_container.markdown(response)

#         message = {"role": "assistant", "content": response}
#         # Parse the response for a SQL query and execute if available
#         sql_match = re.search(r"```sql\n(.*)\n```", response, re.DOTALL)
#         if sql_match:
#             sql = sql_match.group(1)
#             conn = st.experimental_connection("snowpark")
#             message["results"] = conn.query(sql)
#             st.dataframe(message["results"])
#         st.session_state.messages.append(message)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        # Initialize Langchain chain
        # prompt = {"role": "system", "content": get_system_prompt()}

        prompt = ChatPromptTemplate(messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),]
)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        gpt = ChatOpenAI(temperature=0, openai_api_key=st.secrets.OPENAI_API_KEY, model_name= 'gpt-4-1106-preview')
        llm_chain = LLMChain(llm=gpt, prompt=prompt, verbose=True, memory=memory)  # adjust the model as necessary

        # Process the input using Langchain chain
        response = llm_chain.run([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])

        # Display the response
        resp_container = st.empty()
        resp_container.markdown(response)

        message = {"role": "assistant", "content": response}
        # Parse the response for a SQL query and execute if available
        # ... existing SQL processing logic ...

        # Parse the response for a SQL query and execute if available
        sql_match = re.search(r"```sql\n(.*)\n```", response, re.DOTALL)
        if sql_match:
            sql = sql_match.group(1)
            conn = st.experimental_connection("snowpark")
            sql_results = conn.query(sql)
            message["results"] = sql_results
            st.dataframe(message["results"])

            query_results_str = str(sql_results)

            sql_results_message = {"role": "assistant", "content": f"SQL Query Results: {query_results_str}"}

            # Append the SQL results message to the conversation history
            st.session_state.messages.append(sql_results_message)

            # Update the memory with the new message (if your memory implementation supports this)
            
        
        st.session_state.messages.append(message)
        # prepared_messages = []
        # for m in st.session_state.messages:
        #     content_str = str(m["content"]) if m["content"] is not None else ""
        #     prepared_messages.append({"role": m["role"], "content": content_str})

        # response = llm_chain.run(prepared_messages)
        # #response = llm_chain.run([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])