import streamlit as st

#QUALIFIED_TABLE_NAME = "FROSTY_SAMPLE.CYBERSYN_FINANCIAL.FINANCIAL_ENTITY_ANNUAL_TIME_SERIES"

# QUALIFIED_TABLE_NAME = "YOUTUBE_LLM.STAGING.YOUTUBE_LLM_VIDEOS"

# TABLE_DESCRIPTION = """
# This table has various metrics for Youtube channels (main stream media News channels) for the past month .
# The user may describe the entities interchangeably as video views, performance, most watched.
# """


# TABLE_DESCRIPTION = """
# This table has various metrics for financial entities (also referred to as banks) since 1983.
# The user may describe the entities interchangeably as banks, financial institutions, or financial entities.
# """
# This query is optional if running Frosty on your own table, especially a wide table.
# Since this is a deep table, it's useful to tell Frosty what variables are available.
# Similarly, if you have a table with semi-structured data (like JSON), it could be used to provide hints on available keys.
# If altering, you may also need to modify the formatting logic in get_table_context() below.

#METADATA_QUERY = "SELECT VARIABLE_NAME, DEFINITION FROM FROSTY_SAMPLE.CYBERSYN_FINANCIAL.FINANCIAL_ENTITY_ATTRIBUTES_LIMITED;"
# METADATA_QUERY = ""

GEN_SQL = """
You will be acting as an AI Snowflake SQL Expert named YOUTUBE BOT.
Your goal is to give correct, executable sql query to users.
You will be replying to users who will be confused if you don't respond in the character of YOUTUBE BOT.
You are given one table, the table name is in <tableName> tag, the columns are in <columns> tag.
The user will ask questions, for each question you should respond and include a sql query based on the question and the table.
USE DATABASE "YOUTUBE_LLM" 

{context}

Here are 6 critical rules for the interaction you must abide:
<rules>
1. You MUST MUST wrap the generated sql code within ``` sql code markdown in this format e.g
```sql
(select 1) union (select 2)
```
2. If I don't tell you to find a limited set of results in the sql query or question, you MUST limit the number of responses to 10.
3. Text / string where clauses must be fuzzy match e.g ilike %keyword%
4. Make sure to generate a single snowflake sql code, not multiple. 
5. You should only use the table columns given in <columns>, and the table given in <tableName>, you MUST NOT hallucinate about the table names
6. DO NOT put numerical at the very front of sql variable.
</rules>
7. Whenever writing a table name preceed it by databasename eg: YOUTUBE_LLM.table_name

Don't forget to use "ilike %keyword%" for fuzzy match queries (especially for variable_name column)
and wrap the generated sql code with ``` sql code markdown in this format e.g:
```sql
(select 1) union (select 2)
```

For each question from the user, make sure to include a query in your response.

Now to get started, please briefly introduce yourself, describe the table at a high level, and share the available metrics in 2-3 sentences.
Then provide 3 example questions using bullet points.
"""

# @st.cache_data(show_spinner=False)
# def get_table_context(table_name: str, table_description: str, metadata_query: str = None):
#     table = table_name.split(".")
#     conn = st.experimental_connection("snowpark")
#     columns = conn.query(f"""
#         SELECT COLUMN_NAME, DATA_TYPE FROM {table[0].upper()}.INFORMATION_SCHEMA.COLUMNS
#         WHERE TABLE_SCHEMA = '{table[1].upper()}' AND TABLE_NAME = '{table[2].upper()}'
#         """,
#     )
#     columns = "\n".join(
#         [
#             f"- **{columns['COLUMN_NAME'][i]}**: {columns['DATA_TYPE'][i]}"
#             for i in range(len(columns["COLUMN_NAME"]))
#         ]
#     )
#     context = f"""
# Here is the table name <tableName> {'.'.join(table)} </tableName>

# <tableDescription>{table_description}</tableDescription>

# Here are the columns of the {'.'.join(table)}

# <columns>\n\n{columns}\n\n</columns>
#     """
#     if metadata_query:
#         metadata = conn.query(metadata_query)
#         metadata = "\n".join(
#             [
#                 f"- **{metadata['VARIABLE_NAME'][i]}**: {metadata['DEFINITION'][i]}"
#                 for i in range(len(metadata["VARIABLE_NAME"]))
#             ]
#         )
#         context = context + f"\n\nAvailable variables by VARIABLE_NAME:\n\n{metadata}"
#     return context





# def get_system_prompt():
#     table_context = get_table_context(
#         table_name=QUALIFIED_TABLE_NAME,
#         table_description=TABLE_DESCRIPTION,
#         metadata_query=METADATA_QUERY
#     )
#     return GEN_SQL.format(context=table_context)

# # do `streamlit run prompts.py` to view the initial system prompt in a Streamlit app
# if __name__ == "__main__":
#     st.header("System prompt for YOUTUBE BOT")
#     st.markdown(get_system_prompt())


def get_single_table_context(conn, schema_name, database_name, table_name, table_description):
    columns = conn.query(f"""
        SELECT COLUMN_NAME, DATA_TYPE FROM {database_name.upper()}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{schema_name.upper()}' AND TABLE_NAME = '{table_name.upper()}'
        """,
    )
    columns = "\n".join(
        [f"- **{col['COLUMN_NAME']}**: {col['DATA_TYPE']}" for _, col in columns.iterrows()]
    )
    context = f"""
Here is the table name <tableName> {database_name}.{schema_name}.{table_name} </tableName>

<tableDescription>{table_description}</tableDescription>

Here are the columns of {database_name}.{schema_name}.{table_name}

<columns>\n\n{columns}\n\n</columns>
    """
    return context

# Function to get context for all tables
@st.cache_data(show_spinner=False)
def get_all_tables_context(schema_name: str, database_name: str):
    conn = st.experimental_connection("snowpark")
    #conn.query("USE DATABASE YOUTUBE_LLM")
    tables = conn.query(f"SELECT TABLE_NAME FROM {database_name}.INFORMATION_SCHEMA.TABLES where table_schema = '{schema_name}';")
    #tables = conn.query(f"SELECT TABLE_NAME FROM YOUTUBE_LLM.{schema_name}.INFORMATION_SCHEMA.TABLES")

    all_contexts = []
    for _, row in tables.iterrows():
        table_name = row['TABLE_NAME']
        # Here you can dynamically determine the table description or pass a generic one
        table_description = "Description for " + table_name
        table_context = get_single_table_context(conn, schema_name, database_name, table_name, table_description)
        all_contexts.append(table_context)

    return "\n\n".join(all_contexts)

def get_system_prompt():
    all_tables_context = get_all_tables_context(schema_name="STAGING", database_name="YOUTUBE_LLM")
    return GEN_SQL.format(context=all_tables_context)

if __name__ == "__main__":
    st.header("System prompt for YOUTUBE BOT")
    st.markdown(get_system_prompt())

