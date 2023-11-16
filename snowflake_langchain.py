import snowflake.connector
import langchain.chains
import streamlit as st

from langchain.chains import LLMChain
from langchain.llms import OpenAI  
from langchain.utilities import SQLDatabase 
from langchain.chains import create_sql_query_chain
from snowflake.snowpark import Session

# Connect to Snowflake
# conn = snowflake.connector.connect(
#     User='ABBASFURNITUREWALA',
#     password='Abba$123',
#     account='lqeuurc-npb19527',
#     warehouse='COMPUTE_WH',
#     database='YOUTUBE_LLM',
#     schema='RAW'
# )
user='ABBASFURNITUREWALA'
password='Abba$123'
account='lqeuurc-npb19527'
warehouse='COMPUTE_WH'
database='YOUTUBE_LLM'
schema='RAW'
role = "ACCOUNTADMIN"

connection_parameters = { 
    "user": user,
    "password": password,
    "account": account,
    "warehouse": warehouse,
    "database": database,
    "schema": schema,
    "role": role
}


session = Session.builder.configs(connection_parameters).create()

if not connection_parameters.get("user"):
    raise ValueError("Snowflake user is not specified in connection parameters")

session = Session.builder.configs(connection_parameters).create()


# Define a custom component to interact with Snowflake
class SnowflakeComponent:
    def __init__(self, connection):
        self.connection = connection

    def query(self, sql_query):
        # Implement query execution and return results
        pass

# Initialize your components
#snowflake_component = SnowflakeComponent(conn)

gpt = OpenAI(temperature=0, openai_api_key=st.secrets.OPENAI_API_KEY)  # Replace with your preferred LLM if different


snowflake_url = f"snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}&role={role}"

#snowflake_url = f"snowflake://{user}:{password}@{account}/{warehouse}/{database}/{schema}?role={role}"


db = SQLDatabase.from_uri(snowflake_url,sample_rows_in_table_info=1)

# we can see what information is passed to the LLM regarding the database
#print(db.table_info)

  
database_chain = create_sql_query_chain(gpt,db)


prompt = "which channel posts the most videos? "

sql_query = database_chain.invoke({"question": prompt})

if sql_query.endswith(';'):
    sql_query = sql_query[:-1]

#sql_query = "SELECT video_id, title, viewcount FROM videos ORDER BY viewcount DESC LIMIT 5"

#we can visualize what sql query is generated by the LLM
print(sql_query)

session.sql(sql_query).show()


# Create a chain that includes the Snowflake component and the language model
#chain = LLMChain({ llm: gpt, prompt : response })

# # Define a function to handle user input and generate a response
# def handle_query(user_input):
#     # Process the input, interact with Snowflake, generate a response
#     pass

# # Example usage
# response = handle_query("Show recent sales data")

# print(response)


