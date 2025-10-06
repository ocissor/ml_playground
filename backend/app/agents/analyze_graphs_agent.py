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
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import numpy as np
from scipy import stats

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

class PandasQuery(BaseModel):
    query: str = Field(..., description="A valid pandas query string to filter the DataFrame")

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"

# ==================== VISUALIZATION TOOLS ==================== #

@tool
def plot_histogram(uuid: str, column: str, bins: int = 30) -> str:
    '''Plot and return a histogram of a column with customizable bins.'''
    try:
        df = pd.read_json(StringIO(dataframe[uuid]))
        if column not in df.columns:
            return f"Column '{column}' not found in dataset."
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Column '{column}' is not numeric. Cannot create histogram."
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df[column].hist(bins=bins, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title(f"Distribution of {column}", fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean and median lines
        mean_val = df[column].mean()
        median_val = df[column].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.legend()
        
        return fig_to_base64(fig)
    except Exception as e:
        return f"Error creating histogram: {str(e)}"

@tool
def plot_scatter(uuid: str, x: str, y: str, color_by: str = None) -> str:
    '''Plot and return a scatter plot of two columns, optionally colored by a third column.'''
    try:
        df = pd.read_json(StringIO(dataframe[uuid]))
        
        if x not in df.columns or y not in df.columns:
            return f"One or both columns '{x}', '{y}' not found."
        
        # Select only numeric columns
        if not pd.api.types.is_numeric_dtype(df[x]) or not pd.api.types.is_numeric_dtype(df[y]):
            return f"Both columns must be numeric for scatter plot."
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if color_by and color_by in df.columns:
            scatter = ax.scatter(df[x], df[y], c=df[color_by], cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            ax.scatter(df[x], df[y], alpha=0.6, s=50, color='steelblue')
        
        ax.set_title(f"Scatter Plot: {y} vs {x}", fontsize=14, fontweight='bold')
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df[x].dropna(), df[y].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(df[x], p(df[x]), "r--", alpha=0.8, linewidth=2, label='Trend')
        ax.legend()
        
        return fig_to_base64(fig)
    except Exception as e:
        return f"Error creating scatter plot: {str(e)}"

@tool
def plot_heatmap(uuid: str, method: str = "pearson") -> str:
    '''Plot and return a heatmap of the correlation matrix using pearson, spearman, or kendall method.'''
    try:
        df = pd.read_json(StringIO(dataframe[uuid]))
        numeric_df = df.select_dtypes(include='number')
        
        if numeric_df.shape[1] < 2:
            return "Need at least 2 numeric columns for correlation heatmap."
        
        corr = numeric_df.corr(method=method)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap="RdBu_r", center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f"Correlation Heatmap ({method.capitalize()})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        return f"Error creating heatmap: {str(e)}"

@tool
def plot_boxplot(uuid: str, columns: str) -> str:
    '''Create boxplots for specified columns (comma-separated). Useful for outlier detection.'''
    try:
        df = pd.read_json(StringIO(dataframe[uuid]))
        col_list = [c.strip() for c in columns.split(',')]
        
        # Filter to existing numeric columns
        numeric_cols = df.select_dtypes(include='number').columns
        valid_cols = [c for c in col_list if c in numeric_cols]
        
        if not valid_cols:
            return f"No valid numeric columns found from: {columns}"
        
        fig, ax = plt.subplots(figsize=(max(10, len(valid_cols) * 2), 6))
        df[valid_cols].boxplot(ax=ax)
        ax.set_title("Box Plot Analysis", fontsize=14, fontweight='bold')
        ax.set_ylabel("Values", fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        return f"Error creating boxplot: {str(e)}"

@tool
def plot_line(uuid: str, x: str, y: str) -> str:
    '''Create a line plot for time series or sequential data.'''
    try:
        df = pd.read_json(StringIO(dataframe[uuid]))
        
        if x not in df.columns or y not in df.columns:
            return f"One or both columns '{x}', '{y}' not found."
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df[x], df[y], marker='o', linewidth=2, markersize=4, alpha=0.7)
        ax.set_title(f"Line Plot: {y} over {x}", fontsize=14, fontweight='bold')
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        return f"Error creating line plot: {str(e)}"

@tool
def plot_bar(uuid: str, column: str, top_n: int = 10) -> str:
    '''Create a bar chart showing value counts for a categorical column (top N values).'''
    try:
        df = pd.read_json(StringIO(dataframe[uuid]))
        
        if column not in df.columns:
            return f"Column '{column}' not found."
        
        value_counts = df[column].value_counts().head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f"Top {top_n} Values in {column}", fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig_to_base64(fig)
    except Exception as e:
        return f"Error creating bar chart: {str(e)}"

# ==================== ANALYSIS TOOLS ==================== #

@tool
def summarize_column(uuid: str, column: str) -> str:
    '''Return comprehensive summary statistics of a column.'''
    try:
        df = pd.read_json(StringIO(dataframe[uuid]))
        if column not in df.columns:
            return f"Column '{column}' not found."
        
        col_data = df[column]
        summary = [f"Summary statistics for '{column}':"]
        summary.append(f"Data type: {col_data.dtype}")
        summary.append(f"Total values: {len(col_data)}")
        summary.append(f"Missing values: {col_data.isnull().sum()} ({col_data.isnull().sum()/len(col_data)*100:.2f}%)")
        summary.append(f"Unique values: {col_data.nunique()}")
        
        if pd.api.types.is_numeric_dtype(col_data):
            desc = col_data.describe()
            summary.append(f"\nNumeric Statistics:")
            summary.append(f"  Mean: {desc['mean']:.4f}")
            summary.append(f"  Median: {desc['50%']:.4f}")
            summary.append(f"  Std Dev: {desc['std']:.4f}")
            summary.append(f"  Min: {desc['min']:.4f}")
            summary.append(f"  Max: {desc['max']:.4f}")
            summary.append(f"  Q1 (25%): {desc['25%']:.4f}")
            summary.append(f"  Q3 (75%): {desc['75%']:.4f}")
            summary.append(f"  Skewness: {col_data.skew():.4f}")
            summary.append(f"  Kurtosis: {col_data.kurtosis():.4f}")
        else:
            summary.append(f"\nCategorical Statistics:")
            top_values = col_data.value_counts().head(5)
            summary.append("  Top 5 values:")
            for val, count in top_values.items():
                summary.append(f"    {val}: {count} ({count/len(col_data)*100:.2f}%)")
        
        return "\n".join(summary)
    except Exception as e:
        return f"Error summarizing column: {str(e)}"

@tool
def correlation(uuid: str, col1: str, col2: str, method: str = "pearson") -> str:
    '''Calculate and return the correlation between two numeric columns using pearson, spearman, or kendall method.'''
    try:
        df = pd.read_json(StringIO(dataframe[uuid]))
        numeric_df = df.select_dtypes(include='number')
        
        if col1 not in numeric_df.columns or col2 not in numeric_df.columns:
            return f"One or both columns '{col1}', '{col2}' not found or not numeric."
        
        corr_value = numeric_df[col1].corr(numeric_df[col2], method=method)
        
        # Interpretation
        abs_corr = abs(corr_value)
        if abs_corr > 0.7:
            strength = "strong"
        elif abs_corr > 0.4:
            strength = "moderate"
        elif abs_corr > 0.2:
            strength = "weak"
        else:
            strength = "very weak"
        
        direction = "positive" if corr_value > 0 else "negative"
        
        return f"Correlation between '{col1}' and '{col2}' ({method}): {corr_value:.4f}\nInterpretation: {strength} {direction} correlation"
    except Exception as e:
        return f"Error calculating correlation: {str(e)}"

@tool
def detect_outliers(uuid: str, column: str) -> str:
    '''Detect outliers in a numeric column using IQR method and return detailed analysis.'''
    try:
        df = pd.read_json(StringIO(dataframe[uuid]))
        
        if column not in df.columns:
            return f"Column '{column}' not found."
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Column '{column}' is not numeric."
        
        col_data = df[column].dropna()
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        
        result = [f"Outlier Analysis for '{column}':"]
        result.append(f"Total outliers detected: {len(outliers)} ({len(outliers)/len(col_data)*100:.2f}%)")
        result.append(f"Lower bound: {lower_bound:.4f}")
        result.append(f"Upper bound: {upper_bound:.4f}")
        result.append(f"Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
        
        if len(outliers) > 0:
            result.append(f"\nOutlier values (showing first 10):")
            for val in outliers.head(10):
                result.append(f"  {val:.4f}")
        
        return "\n".join(result)
    except Exception as e:
        return f"Error detecting outliers: {str(e)}"

@tool
def query_data(uuid: str, query: str) -> str:
    """
    Filter DataFrame using natural language query converted to pandas query string.
    Examples: 'Age > 30', 'Price < 100 and Category == "Electronics"'
    """
    try:
        template_str = """
        You are an assistant that converts natural language questions about a pandas DataFrame into valid pandas query strings.
        Only output the query string without any explanation or markdown formatting.

        Question: {user_input}
        
        Examples:
        - "rows where age is greater than 30" -> "Age > 30"
        - "items with price less than 100" -> "Price < 100"
        - "records where status is active" -> "Status == 'active'"
        """

        prompt = PromptTemplate.from_template(template_str)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0) 
        structured_llm = llm.with_structured_output(PandasQuery)
        query_chain = prompt | structured_llm

        df = pd.read_json(StringIO(dataframe[uuid]))
        result = query_chain.invoke({'user_input': query})
        pandas_query = result.query
        
        filtered_df = df.query(pandas_query)
        
        if filtered_df.empty:
            return f"Query '{pandas_query}' returned no results."
        
        result_str = [f"Query: {pandas_query}"]
        result_str.append(f"Returned {len(filtered_df)} rows out of {len(df)} total rows")
        result_str.append(f"\nFirst 10 results:\n{filtered_df.head(10).to_string()}")
        
        return "\n".join(result_str)
    except Exception as e:
        return f"Error executing query: {str(e)}"

@tool
def get_data_info(uuid: str) -> str:
    '''Get comprehensive information about the dataset including shape, columns, types, and missing values.'''
    try:
        df = pd.read_json(StringIO(dataframe[uuid]))
        
        info = [f"Dataset Information:"]
        info.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        info.append(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        info.append(f"\nColumns and Data Types:")
        
        for col in df.columns:
            dtype = df[col].dtype
            missing = df[col].isnull().sum()
            unique = df[col].nunique()
            info.append(f"  {col}: {dtype} | Missing: {missing} | Unique: {unique}")
        
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        
        info.append(f"\nNumeric columns ({len(numeric_cols)}): {', '.join(numeric_cols)}")
        info.append(f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols)}")
        info.append(f"Total missing values: {df.isnull().sum().sum()}")
        info.append(f"Duplicate rows: {df.duplicated().sum()}")
        
        return "\n".join(info)
    except Exception as e:
        return f"Error getting data info: {str(e)}"

@tool
def compare_groups(uuid: str, group_column: str, value_column: str) -> str:
    '''Compare statistics of a numeric column across different groups of a categorical column.'''
    try:
        df = pd.read_json(StringIO(dataframe[uuid]))
        
        if group_column not in df.columns or value_column not in df.columns:
            return f"One or both columns not found."
        
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            return f"Value column '{value_column}' must be numeric."
        
        grouped = df.groupby(group_column)[value_column].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(4)
        
        result = [f"Group comparison: {value_column} by {group_column}\n"]
        result.append(grouped.to_string())
        
        # Add ANOVA test if more than 2 groups
        groups = [group[value_column].dropna() for name, group in df.groupby(group_column)]
        if len(groups) > 2:
            f_stat, p_value = stats.f_oneway(*groups)
            result.append(f"\n\nANOVA Test:")
            result.append(f"F-statistic: {f_stat:.4f}")
            result.append(f"P-value: {p_value:.4f}")
            result.append(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        return "\n".join(result)
    except Exception as e:
        return f"Error comparing groups: {str(e)}"

@tool
def missing_value_analysis(uuid: str) -> str:
    '''Analyze patterns of missing values across all columns in the dataset.'''
    try:
        df = pd.read_json(StringIO(dataframe[uuid]))
        
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            return "No missing values found in the dataset!"
        
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        
        result = [f"Missing Value Analysis:"]
        result.append(f"Total missing cells: {total_missing} out of {total_cells} ({total_missing/total_cells*100:.2f}%)")
        result.append(f"\nMissing values by column:")
        
        for col, count in missing.items():
            percentage = (count / len(df)) * 100
            result.append(f"  {col}: {count} ({percentage:.2f}%)")
        
        # Complete cases
        complete_rows = df.dropna().shape[0]
        result.append(f"\nComplete rows (no missing values): {complete_rows} ({complete_rows/len(df)*100:.2f}%)")
        
        return "\n".join(result)
    except Exception as e:
        return f"Error analyzing missing values: {str(e)}"

@tool
def distribution_test(uuid: str, column: str) -> str:
    '''Test if a numeric column follows a normal distribution using Shapiro-Wilk test.'''
    try:
        df = pd.read_json(StringIO(dataframe[uuid]))
        
        if column not in df.columns:
            return f"Column '{column}' not found."
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Column '{column}' must be numeric."
        
        col_data = df[column].dropna()
        
        if len(col_data) < 3:
            return f"Not enough data points for normality test (need at least 3)."
        
        # Use first 5000 points for performance
        sample = col_data[:5000]
        stat, p_value = stats.shapiro(sample)
        
        result = [f"Normality Test for '{column}' (Shapiro-Wilk):"]
        result.append(f"Sample size: {len(sample)}")
        result.append(f"Test statistic: {stat:.4f}")
        result.append(f"P-value: {p_value:.6f}")
        result.append(f"\nInterpretation:")
        
        if p_value > 0.05:
            result.append(f"  The data appears to be normally distributed (p > 0.05)")
        else:
            result.append(f"  The data does NOT appear to be normally distributed (p < 0.05)")
        
        result.append(f"\nAdditional metrics:")
        result.append(f"  Skewness: {col_data.skew():.4f} (0 = symmetric)")
        result.append(f"  Kurtosis: {col_data.kurtosis():.4f} (0 = normal)")
        
        return "\n".join(result)
    except Exception as e:
        return f"Error testing distribution: {str(e)}"

# ==================== AGENT FUNCTION ==================== #

def data_analyzer_agent(state: GraphState) -> GraphState:
    """Enhanced data analyzer agent with comprehensive tools"""
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)   
    
    # All available tools
    tools = [
        # Visualization tools
        plot_histogram,
        plot_scatter,
        plot_heatmap,
        plot_boxplot,
        plot_line,
        plot_bar,
        # Analysis tools
        summarize_column,
        correlation,
        detect_outliers,
        query_data,
        get_data_info,
        compare_groups,
        missing_value_analysis,
        distribution_test
    ]
    
    llm_with_tool = llm.bind_tools(tools) 

    user_input = state['user_input']
    uuid = state['uuid']
    
    # Validate UUID
    if uuid not in dataframe:
        state['messages'].append(AIMessage(
            content="Error: Dataset not found. Please upload your data first.",
            type="ai"
        ))
        return state
    
    messages = state['messages']
    
    # Enhanced system message with better instructions
    system_message = SystemMessage(content=f"""You are an expert data analysis assistant with access to powerful analytical tools. 

Your capabilities include:
- Creating various visualizations (histograms, scatter plots, heatmaps, box plots, line plots, bar charts)
- Performing statistical analysis (correlations, distributions, outlier detection)
- Querying and filtering data
- Comparing groups and conducting statistical tests
- Analyzing missing values and data quality

The dataset UUID is: {uuid}

Guidelines:
1. Always use the appropriate tool for the user's request
2. Provide clear, insightful interpretations of the results
3. If a visualization is generated, explain what it shows
4. For statistical results, provide context and interpretation
5. Be proactive in suggesting related analyses that might be helpful
6. If the user's request is unclear, use get_data_info first to understand the dataset

Remember to always pass the uuid parameter to every tool call.""")
    
    # Build message list
    messages_list = [system_message] + list(messages) + [HumanMessage(content=user_input)]
    
    # Remove empty messages
    messages_list = [msg for msg in messages_list if msg is not None and msg.content.strip() != ""]
    
    # Get initial response from LLM
    try:
        response = llm_with_tool.invoke(messages_list)
        state['messages'].append(AIMessage(content=response.content if response.content else "", type="ai"))
        
        # Process tool calls if any
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                # Ensure uuid is in arguments
                if 'uuid' not in tool_args:
                    tool_args['uuid'] = uuid
                
                # Find and execute the tool
                tool = next((t for t in tools if vars(t)['name'] == tool_name), None)
                
                if tool:
                    try:
                        tool_response = tool.invoke(tool_args)
                        
                        # Add tool response to messages
                        state['messages'].append(ToolMessage(
                            content=tool_response,
                            type="tool",
                            tool_name=tool_name,
                            tool_call_id=tool_call['id']
                        ))
                        
                        # Generate final response based on tool output
                        if tool_response.startswith("data:image/png;base64,"):
                            follow_up = HumanMessage(
                                content=f"The tool '{tool_name}' has generated a visualization. Please provide a brief interpretation of what this plot shows and any insights."
                            )
                        elif tool_response.startswith("Error"):
                            follow_up = HumanMessage(
                                content=f"The tool encountered an error: {tool_response}. Please explain this to the user and suggest alternatives."
                            )
                        else:
                            follow_up = HumanMessage(
                                content=f"Based on these results, provide a clear and concise interpretation for the user. Highlight key findings and insights."
                            )
                        
                        # Get final interpretation
                        interpretation_messages = state['messages'] + [follow_up]
                        final_response = llm.invoke(interpretation_messages)
                        
                        state['messages'].append(AIMessage(
                            content=final_response.content,
                            type="ai"
                        ))
                        
                    except Exception as e:
                        error_msg = f"Error executing tool {tool_name}: {str(e)}"
                        state['messages'].append(AIMessage(
                            content=error_msg,
                            type="ai"
                        ))
                else:
                    state['messages'].append(AIMessage(
                        content=f"Tool '{tool_name}' not found.",
                        type="ai"
                    ))
        
    except Exception as e:
        state['messages'].append(AIMessage(
            content=f"An error occurred: {str(e)}. Please try rephrasing your question.",
            type="ai"
        ))
    
    # Clean up messages
    state['messages'] = [msg for msg in state['messages'] if msg is not None and msg.content.strip() != ""]
    
    return state