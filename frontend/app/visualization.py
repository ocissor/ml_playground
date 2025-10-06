import streamlit as st
import requests
import json
import base64
from io import BytesIO  
from PIL import Image
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def viz_data(app_state):
    
    st.set_page_config(page_title="ML Playground 🚀", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3em;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 10px 0;
        }
        .insight-box {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .chat-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🚀 ML Playground - Advanced Analytics")
    st.markdown("### Explore, Analyze, and Chat with Your Data")
    
    # ------------------- Data Upload Info ------------------- #
    if not app_state.uploaded_file:
        st.info("👈 Please upload a dataset from the sidebar to get started!")
        st.markdown("""
        ### 🎯 What You Can Do:
        - **📊 Explore Data**: Get comprehensive statistics and insights
        - **📈 Visualizations**: Interactive charts and plots
        - **🤖 AI Chat**: Ask questions about your data in natural language
        - **🔍 Advanced Analysis**: Correlation matrices, distributions, and more
        - **💡 Smart Insights**: AI-generated insights about your dataset
        """)
        return
    
    # ------------------- Tabs for Organization ------------------- #
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "🤖 AI Chat", "🔬 Advanced Analysis"])
    
    # ==================== TAB 1: OVERVIEW ==================== #
    with tab1:
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        # Quick Stats Cards
        try:
            app_state.uploaded_file.seek(0)
            df = pd.read_csv(app_state.uploaded_file)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h2>{df.shape[0]:,}</h2>
                        <p>Total Rows</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <h2>{df.shape[1]}</h2>
                        <p>Total Columns</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
                st.markdown(f"""
                    <div class="metric-card">
                        <h2>{missing_pct:.1f}%</h2>
                        <p>Missing Data</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Data Preview
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 🔍 Data Preview")
                st.dataframe(df.head(10), use_container_width=True, height=300)
            
            with col2:
                st.markdown("#### 📋 Column Types")
                dtype_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Type': df.dtypes.values.astype(str)
                })
                st.dataframe(dtype_df, use_container_width=True, height=300)
            
            st.markdown("---")
            
            # Detailed Statistics
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("📊 Get Detailed Statistics", use_container_width=True):
                    with st.spinner("Calculating statistics..."):
                        try:
                            app_state.uploaded_file.seek(0)
                            response = requests.post(
                                "http://ml-playground-backend-service:8000/data_stats",
                                files={"file": app_state.uploaded_file}
                            )
                            if response.status_code == 200:
                                response_json = response.json()
                                if 'basic_stats' in response_json:
                                    stats_df = pd.read_json(response_json['basic_stats'])
                                    st.dataframe(stats_df, use_container_width=True)
                                else:
                                    st.info("No data stats available")
                            else:
                                st.error("Could not retrieve data stats")
                        except Exception as e:
                            st.error(f"Request failed: {e}")
            
            with col2:
                if st.button("🎯 Generate AI Insights", use_container_width=True):
                    with st.spinner("Generating insights..."):
                        # Generate quick insights
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            st.markdown("#### 💡 Quick Insights:")
                            
                            # Most correlated features
                            corr_matrix = df[numeric_cols].corr()
                            high_corr = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i+1, len(corr_matrix.columns)):
                                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                                        high_corr.append((
                                            corr_matrix.columns[i],
                                            corr_matrix.columns[j],
                                            corr_matrix.iloc[i, j]
                                        ))
                            
                            if high_corr:
                                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                                st.write("**🔗 Highly Correlated Features:**")
                                for col1, col2, corr in high_corr[:3]:
                                    st.write(f"- {col1} ↔ {col2}: {corr:.3f}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Outlier detection
                            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                            st.write("**⚠️ Outlier Detection:**")
                            for col in numeric_cols[:3]:
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                                if outliers > 0:
                                    st.write(f"- {col}: {outliers} outliers detected")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info("No numeric columns found for insights")
            
            # Missing Data Analysis
            if df.isnull().sum().sum() > 0:
                st.markdown("---")
                st.markdown("#### 🔴 Missing Data Analysis")
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                
                if len(missing_data) > 0:
                    fig = px.bar(
                        x=missing_data.values,
                        y=missing_data.index,
                        orientation='h',
                        labels={'x': 'Missing Count', 'y': 'Column'},
                        title='Missing Values by Column'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    # ==================== TAB 2: VISUALIZATIONS ==================== #
    with tab2:
        st.subheader("📈 Data Visualizations")
        
        viz_type = st.radio(
            "Choose Visualization Type:",
            ["📊 Auto-Generated Plots", "🎨 Custom Plots", "🔥 Heatmaps"],
            horizontal=True
        )
        
        if viz_type == "📊 Auto-Generated Plots":
            if st.button("🎨 Generate All Visualizations", use_container_width=True):
                with st.spinner("Creating beautiful visualizations..."):
                    try:
                        app_state.uploaded_file.seek(0)
                        response = requests.post(
                            "http://ml-playground-backend-service:8000/visualize_data",
                            files={"file": app_state.uploaded_file}
                        )
                        if response.status_code == 200:
                            images = response.json()
                            if images:
                                st.success(f"✅ Generated {len(images)} visualizations!")
                                
                                # Display in a nice grid
                                num_cols = 2
                                cols = st.columns(num_cols)
                                for idx, (title, img_bytes) in enumerate(images.items()):
                                    img = Image.open(BytesIO(base64.b64decode(img_bytes)))
                                    with cols[idx % num_cols]:
                                        st.image(img, caption=title, use_container_width=True)
                            else:
                                st.info("No visualizations available for the provided data.")
                        else:
                            st.error("Could not retrieve visualizations")
                    except Exception as e:
                        st.error(f"Request failed: {e}")
        
        elif viz_type == "🎨 Custom Plots":
            try:
                app_state.uploaded_file.seek(0)
                df = pd.read_csv(app_state.uploaded_file)
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                all_cols = df.columns.tolist()
                
                plot_type = st.selectbox("Select Plot Type", [
                    "Scatter Plot", "Line Plot", "Bar Chart", "Box Plot", "Violin Plot", "Histogram"
                ])
                
                col1, col2 = st.columns(2)
                
                if plot_type in ["Scatter Plot", "Line Plot"]:
                    with col1:
                        x_col = st.selectbox("X-axis", numeric_cols)
                    with col2:
                        y_col = st.selectbox("Y-axis", numeric_cols)
                    
                    if st.button("Generate Plot"):
                        if plot_type == "Scatter Plot":
                            fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                        else:
                            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type in ["Box Plot", "Violin Plot"]:
                    with col1:
                        col = st.selectbox("Select Column", numeric_cols)
                    
                    if st.button("Generate Plot"):
                        if plot_type == "Box Plot":
                            fig = px.box(df, y=col, title=f"Box Plot of {col}")
                        else:
                            fig = px.violin(df, y=col, title=f"Violin Plot of {col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Histogram":
                    with col1:
                        col = st.selectbox("Select Column", numeric_cols)
                    with col2:
                        bins = st.slider("Number of Bins", 10, 100, 30)
                    
                    if st.button("Generate Plot"):
                        fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Bar Chart":
                    with col1:
                        col = st.selectbox("Select Column", all_cols)
                    
                    if st.button("Generate Plot"):
                        value_counts = df[col].value_counts().head(20)
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                   title=f"Top 20 Values in {col}",
                                   labels={'x': col, 'y': 'Count'})
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error: {e}")
        
        else:  # Heatmaps
            st.info("🤖 Use the AI Chat to generate correlation heatmaps and advanced visualizations!")
    
    # ==================== TAB 3: AI CHAT ==================== #
    with tab3:
        st.subheader("🤖 Chat with Your Data")
        
        st.markdown("""
        <div class="insight-box">
        <b>💡 What you can ask:</b><br>
        • "Show me a histogram of Age"<br>
        • "What's the correlation between Price and Sales?"<br>
        • "Generate a heatmap of correlations"<br>
        • "Show me a scatter plot of X vs Y"<br>
        • "Filter data where Age > 30"<br>
        • "Summarize the Revenue column"
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.chat_message("user").markdown(message['content'])
                else:
                    with st.chat_message("assistant"):
                        if message['type'] == 'text':
                            st.markdown(message['content'])
                        elif message['type'] == 'image':
                            st.image(message['content'], use_container_width=True)
        
        # Chat input
        user_input = st.chat_input("Ask any query about your data...", key="data_query")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            st.chat_message("user").markdown(user_input)
            
            with st.spinner("🤔 Analyzing your data..."):
                try:
                    app_state.uploaded_file.seek(0)
                    response = requests.post(
                        "http://ml-playground-backend-service:8000/chat_with_data",
                        files={"file": app_state.uploaded_file},
                        data={"user_input": user_input, 'uuid': app_state.uuid}
                    )
                    
                    if response.status_code == 200:
                        output = pickle.loads(response.content)
                        
                        with st.chat_message("assistant"):
                            for message in reversed(output['messages'][-2:]):
                                if message.type == 'ai':
                                    formatted_content = message.content.replace("\\n", "\n")
                                    st.markdown(formatted_content)
                                    st.session_state.chat_history.append({
                                        'role': 'assistant',
                                        'type': 'text',
                                        'content': formatted_content
                                    })
                                    
                                elif message.type == 'tool':
                                    if message.content.startswith("data:image"):
                                        img_bytes = base64.b64decode(message.content.split(',')[1])
                                        img = Image.open(BytesIO(img_bytes))
                                        st.image(img, use_container_width=True)
                                        st.session_state.chat_history.append({
                                            'role': 'assistant',
                                            'type': 'image',
                                            'content': img
                                        })
                                    else:
                                        formatted_content = message.content.replace("\\n", "\n")
                                        st.text(formatted_content)
                                        st.session_state.chat_history.append({
                                            'role': 'assistant',
                                            'type': 'text',
                                            'content': formatted_content
                                        })
                    else:
                        st.error(f"Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Request failed: {e}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # ==================== TAB 4: ADVANCED ANALYSIS ==================== #
    with tab4:
        st.subheader("🔬 Advanced Analysis")
        
        analysis_type = st.selectbox("Select Analysis Type", [
            "Correlation Matrix",
            "Distribution Analysis",
            "Statistical Tests",
            "Feature Importance Preview"
        ])
        
        try:
            app_state.uploaded_file.seek(0)
            df = pd.read_csv(app_state.uploaded_file)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if analysis_type == "Correlation Matrix":
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix,
                                  labels=dict(color="Correlation"),
                                  x=corr_matrix.columns,
                                  y=corr_matrix.columns,
                                  color_continuous_scale='RdBu_r',
                                  aspect="auto")
                    fig.update_layout(title="Correlation Matrix", height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### 🔗 Strong Correlations")
                    high_corr = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > 0.5:
                                high_corr.append({
                                    'Feature 1': corr_matrix.columns[i],
                                    'Feature 2': corr_matrix.columns[j],
                                    'Correlation': round(corr_matrix.iloc[i, j], 3)
                                })
                    if high_corr:
                        st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
                    else:
                        st.info("No strong correlations found (|r| > 0.5)")
                else:
                    st.warning("Need at least 2 numeric columns for correlation analysis")
            
            elif analysis_type == "Distribution Analysis":
                if numeric_cols:
                    selected_col = st.selectbox("Select Column", numeric_cols)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(df, x=selected_col, marginal="box",
                                         title=f"Distribution of {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    st.markdown("#### 📊 Statistical Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{df[selected_col].mean():.2f}")
                    with col2:
                        st.metric("Median", f"{df[selected_col].median():.2f}")
                    with col3:
                        st.metric("Std Dev", f"{df[selected_col].std():.2f}")
                    with col4:
                        st.metric("Skewness", f"{df[selected_col].skew():.2f}")
                else:
                    st.warning("No numeric columns available")
            
            elif analysis_type == "Statistical Tests":
                st.markdown("""
                <div class="insight-box">
                Use the AI Chat to perform statistical tests like:
                <ul>
                <li>T-tests</li>
                <li>ANOVA</li>
                <li>Chi-square tests</li>
                <li>Normality tests</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            else:  # Feature Importance Preview
                st.info("🎯 Train a model in the Model Training section to see feature importance!")
                
        except Exception as e:
            st.error(f"Error: {e}")