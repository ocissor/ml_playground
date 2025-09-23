from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.app.agents.state import GraphState
import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64   
import io
from io import StringIO
from backend.app.data_storage import dataframe
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"

# Tool: Histogram
@tool
def plot_histogram(uuid: str, column: str) -> str:
    '''Plot and return a histogram of a column.'''
    df = pd.read_json(StringIO(dataframe[uuid]))
    if column not in df.columns:
        return f"Column '{column}' not found."
    fig = plt.figure()
    df[column].hist()
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    return fig_to_base64(fig)

# Tool: Scatter plot
@tool
def plot_scatter(uuid: str, x: str, y: str) -> str:
    '''Plot and return a scatter plot of two columns.'''
    df = pd.read_json(StringIO(dataframe[uuid]))
    # Select only numeric columns
    df = df.select_dtypes(include='number')
    if x not in df.columns or y not in df.columns:
        return f"One or both columns '{x}', '{y}' not found."
    fig = plt.figure()
    plt.scatter(df[x], df[y])
    plt.title(f"Scatter plot of {y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    return fig_to_base64(fig)

# Tool: Heatmap of correlation matrix
@tool
def plot_heatmap(uuid: str) -> str:
    '''Plot and return a heatmap of the correlation matrix.'''
    df = pd.read_json(StringIO(dataframe[uuid]))
    df = df.select_dtypes(include='number')
    corr = df.corr()
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    return fig_to_base64(fig)

# Tool: Summary statistics
@tool
def summarize_column(uuid:str, column: str) -> str:
    '''Return summary statistics of a column.'''
    df = pd.read_json(StringIO(dataframe[uuid]))
    if column not in df.columns:
        return f"Column '{column}' not found."
    desc = df[column].describe().to_string()
    return f"Summary statistics for '{column}':\n{desc}"

# Tool: Correlation between two columns
@tool
def correlation(uuid: str, col1: str, col2: str) -> str:
    '''Calculate and return the correlation between two columns.'''
    df = pd.read_json(StringIO(dataframe[uuid]))
    df = df.select_dtypes(include='number')
    if col1 not in df.columns or col2 not in df.columns:
        return f"One or both columns '{col1}', '{col2}' not found."
    corr_value = df[col1].corr(df[col2])
    return f"Correlation between '{col1}' and '{col2}': {corr_value:.4f}"

# Tool: Query the DataFrame with a pandas query string
@tool
def query_data(uuid: str , query: str) -> str:
    """Run a pandas query on the DataFrame and return the first 10 rows as a string."""
    df = pd.read_json(StringIO(dataframe[uuid]))
    try:
        result = df.query(query)
        if result.empty:
            return "Query returned no results."
        return result.head(10).to_string()
    except Exception as e:
        return f"Error running query: {e}"

def data_analyzer_agent(state : GraphState) -> GraphState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")   
    tools = [plot_histogram, plot_scatter, plot_heatmap, summarize_column, correlation, query_data]
    llm_with_tool = llm.bind_tools(tools) 

    user_input = state['user_input']
    if state['uuid'] not in dataframe:
        raise ValueError("UUID not found in dataframe storage.")
    messages = state['messages']
    uuid = state['uuid']

    system_message = SystemMessage(content = f"You are a data analysis assistant. You have access to the following tools to help analyze the provided dataset the uuid of the dataset is {uuid}")
    messages = [system_message] + list(messages) + [HumanMessage(content=user_input)] 
    messages = [msg for msg in messages if msg is not None and msg.content.strip() != ""]
    response = llm_with_tool.invoke(messages)
    state['messages'] = messages + [AIMessage(content=response.content)]

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            if 'uuid' not in tool_args:
                tool_args['uuid'] = uuid
            tool = next((t for t in tools if vars(t)['name'] == tool_name), None)
            if tool:
                tool_response = tool.invoke(tool_args)
                state['messages'].append(ToolMessage(content=tool_response, type = "tool", tool_name=tool_name, tool_call_id = tool_call['id']))

                if tool_response.startswith("data:image/png;base64,"):
                    new_input = HumanMessage(content=f"The tool '{tool_name}' has generated a plot. The plot is encoded in base64 format.")
                else:
                    new_input = HumanMessage(content="Based on the tool outputs, provide a concise response to the user.")
                
                messages = state['messages'] + [new_input]
                final_response = llm.invoke(messages)
                state['messages'].append(AIMessage(content = final_response.content, type = "ai"))

    state['messages'] = [msg for msg in state['messages'] if msg is not None and msg.content.strip() != ""]
    
    return state








