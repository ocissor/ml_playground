from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver
from backend.app.agents.analyze_graphs_agent import data_analyzer_agent
from backend.app.agents.state import GraphState


analyze_data_graph = StateGraph(GraphState)

analyze_data_graph.add_node("data_analyzer",data_analyzer_agent)

analyze_data_graph.add_edge(START, "data_analyzer")

analyze_data_graph.add_edge("data_analyzer", END)

checkpointer = InMemorySaver()
analyze_data_graph = analyze_data_graph.compile(checkpointer=checkpointer)

# import pandas as pd

# df = pd.DataFrame({
#     "name": ["Alice", "Bob", "Charlie", "Diana"],
#     "age": [25, 30, 35, 40],
#     "city": ["New York", "London", "Paris", "Tokyo"],
#     "salary": [70000, 80000, 90000, 100000]
# })

# # Convert to JSON string
# json_str = df.to_json(orient="records")


# if __name__ == "__main__":
#     # Example usage

#     config = {"configurable": {"thread_id": "1"}}
#     input = {"messages": [], "user_input": "summarize the column age", "data": json_str, "uuid": "1"}
#     out = analyze_data_graph.invoke(input, config)
    


