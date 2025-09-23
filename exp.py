from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
import pandas as pd
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
# Load your DataFrame
df = pd.read_csv("data/iris.csv")  # Replace with your data source

# Define a Pydantic model for the expected output
class PandasQuery(BaseModel):
    query: str = Field(description="A valid pandas query string to filter the DataFrame")

# Create a prompt template instructing the LLM to produce a pandas query string
template_str = """
You are an assistant that converts natural language questions about a pandas DataFrame into valid pandas query strings.
Only output the query string without any explanation.

Question: {user_input}
"""

# Create a PromptTemplate with one input variable
prompt = PromptTemplate.from_template(template_str)

# Initialize the LLM with structured output
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0) 
structured_llm = llm.with_structured_output(PandasQuery)

# Compose a runnable chain: passthrough input -> prompt -> structured LLM
query_chain = prompt | structured_llm

def run_pandas_query(nl_query: str) -> str:
    # Get the pandas query string from the LLM
    result = query_chain.invoke({'user_input':nl_query})
    pandas_query = result.query
    print(f"Generated pandas query: {pandas_query}")
    try:
        filtered_df = df.query(pandas_query)
        if filtered_df.empty:
            return "No results found for the query."
        return filtered_df.head(10).to_string()
    except Exception as e:
        return f"Error executing query: {e}"

# Example usage
nl_question = "Show me all entries where the species is 'setosa' and the petal length is greater than 1.5"
print(run_pandas_query(nl_question))
