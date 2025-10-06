import streamlit as st
import requests
import uuid as uuid_lib
from io import BytesIO
import pandas as pd

# Import page modules
from frontend.app.visualization import viz_data
from frontend.app.train import train_models  # You'll need to create this

class AppState:
    """Centralized application state management"""
    def __init__(self):
        if 'uuid' not in st.session_state:
            st.session_state.uuid = str(uuid_lib.uuid4())
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
        if 'file_uploaded' not in st.session_state:
            st.session_state.file_uploaded = False
        if 'dataset_info' not in st.session_state:
            st.session_state.dataset_info = None
    
    @property
    def uuid(self):
        return st.session_state.uuid
    
    @property
    def uploaded_file(self):
        return st.session_state.uploaded_file
    
    @uploaded_file.setter
    def uploaded_file(self, value):
        st.session_state.uploaded_file = value
    
    @property
    def file_uploaded(self):
        return st.session_state.file_uploaded
    
    @file_uploaded.setter
    def file_uploaded(self, value):
        st.session_state.file_uploaded = value
    
    @property
    def dataset_info(self):
        return st.session_state.dataset_info
    
    @dataset_info.setter
    def dataset_info(self, value):
        st.session_state.dataset_info = value

def main():
    # Initialize app state
    app_state = AppState()
    
    # Page configuration
    st.set_page_config(
        page_title="ML Playground 🚀",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for beautiful UI
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            padding: 0rem 1rem;
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        [data-testid="stSidebar"] .css-1d391kg {
            color: white;
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            height: 3em;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .upload-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        
        .info-card {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        .success-banner {
            background: linear-gradient(90deg, #00b09b 0%, #96c93d 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            margin: 20px 0;
        }
        
        h1, h2, h3 {
            font-weight: 800;
        }
        
        .dataset-stats {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }
        
        .stat-box {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex: 1;
            margin: 0 10px;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: 800;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🚀 ML Playground")
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📍 Navigation")
        # page = st.radio(
        #     "Select Page",
        #     ["📊 Data Exploration", "🤖 Model Training", "📈 Model Comparison", "ℹ️ About"],
        #     label_visibility="collapsed"
        # )

        page = st.radio(
            "Select Page",
            ["📊 Data Exploration", "🤖 Model Training", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # File Upload Section
        st.markdown("### 📂 Upload Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset in CSV format"
        )
        
        if uploaded_file is not None:
            # Check if this is a new file
            if app_state.uploaded_file is None or uploaded_file.name != app_state.uploaded_file.name:
                app_state.uploaded_file = uploaded_file
                app_state.file_uploaded = False
            
            # Upload button
            if not app_state.file_uploaded:
                if st.button("🚀 Upload & Process", type="primary"):
                    with st.spinner("Uploading and processing..."):
                        try:
                            # Read file to get info
                            app_state.uploaded_file.seek(0)
                            df = pd.read_csv(app_state.uploaded_file)
                            
                            # Upload to backend
                            app_state.uploaded_file.seek(0)
                            response = requests.post(
                                "http://ml-playground-backend-service:8000/upload_file",
                                files={"file": app_state.uploaded_file},
                                data={"uuid": app_state.uuid}
                            )
                            
                            if response.status_code == 200:
                                app_state.file_uploaded = True
                                app_state.dataset_info = {
                                    'name': uploaded_file.name,
                                    'rows': df.shape[0],
                                    'columns': df.shape[1],
                                    'size': uploaded_file.size / 1024  # KB
                                }
                                st.success("✅ File uploaded successfully!")
                                st.rerun()
                            else:
                                st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.success("✅ File uploaded!")
                
                # Show dataset info
                if app_state.dataset_info:
                    st.markdown("#### 📊 Dataset Info")
                    st.markdown(f"**Name:** {app_state.dataset_info['name']}")
                    st.markdown(f"**Rows:** {app_state.dataset_info['rows']:,}")
                    st.markdown(f"**Columns:** {app_state.dataset_info['columns']}")
                    st.markdown(f"**Size:** {app_state.dataset_info['size']:.2f} KB")
                
                # Clear data button
                if st.button("🗑️ Clear Dataset"):
                    try:
                        requests.delete(f"http://ml-playground-backend-service:8000/clear_session/{app_state.uuid}")
                        app_state.uploaded_file = None
                        app_state.file_uploaded = False
                        app_state.dataset_info = None
                        st.session_state.uuid = str(uuid_lib.uuid4())
                        st.success("Dataset cleared!")
                        st.rerun()
                    except:
                        pass
        
        st.markdown("---")
        
        # Quick Stats
        if app_state.file_uploaded and app_state.dataset_info:
            st.markdown("### 📈 Quick Stats")
            st.metric("Total Records", f"{app_state.dataset_info['rows']:,}")
            st.metric("Features", app_state.dataset_info['columns'])
        
        st.markdown("---")
        
        # Help Section
        with st.expander("❓ Help & Tips"):
            st.markdown("""
            **Getting Started:**
            1. Upload a CSV file
            2. Click 'Upload & Process'
            3. Explore your data!
            
            **Features:**
            - Interactive visualizations
            - AI-powered chat
            - Model training
            - Statistical analysis
            
            **Tips:**
            - Ensure your CSV is well-formatted
            - Check for missing values
            - Use the AI chat for insights
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: white; font-size: 0.8em;'>
                Made with ❤️ using Streamlit<br>
                Version 2.0.0
            </div>
        """, unsafe_allow_html=True)
    
    # Main Content Area
    if page == "📊 Data Exploration":
        viz_data(app_state)
    
    elif page == "🤖 Model Training":
        if app_state.file_uploaded:
            train_models(app_state)
        else:
            st.title("🤖 Model Training")
            st.info("👈 Please upload a dataset from the sidebar to start training models!")
            
            st.markdown("""
            ### 🎯 What You Can Do:
            - **Train Multiple Models**: Classification and Regression
            - **Hyperparameter Tuning**: Optimize model performance
            - **Model Comparison**: Compare different algorithms
            - **Feature Importance**: Understand which features matter most
            - **Model Export**: Download trained models
            
            ### 📚 Supported Algorithms:
            #### Classification:
            - Logistic Regression
            - Random Forest
            - Support Vector Machine (SVM)
            - Gradient Boosting
            - XGBoost
            - Neural Networks
            
            #### Regression:
            - Linear Regression
            - Ridge & Lasso
            - Random Forest
            - Gradient Boosting
            - XGBoost
            """)
    
    # elif page == "📈 Model Comparison":
    #     st.title("📈 Model Comparison Dashboard")
    #     if app_state.file_uploaded:
    #         st.info("🚧 Model comparison feature coming soon!")
    #         st.markdown("""
    #         This section will allow you to:
    #         - Compare multiple trained models
    #         - View performance metrics side-by-side
    #         - Generate comparison visualizations
    #         - Export comparison reports
    #         """)
    #     else:
    #         st.info("👈 Please upload a dataset to get started!")
    
    elif page == "ℹ️ About":
        st.title("ℹ️ About ML Playground")
        
        st.markdown("""
        ## Welcome to ML Playground! 🚀
        
        ML Playground is a comprehensive machine learning platform designed to make data analysis
        and model training accessible and interactive.
        
        ### ✨ Key Features:
        
        #### 📊 Data Exploration
        - **Interactive Visualizations**: Generate beautiful charts and plots
        - **Statistical Analysis**: Comprehensive data profiling
        - **AI-Powered Chat**: Ask questions about your data in natural language
        - **Missing Data Analysis**: Identify and understand data quality issues
        - **Correlation Analysis**: Discover relationships between features
        
        #### 🤖 Model Training
        - **Multiple Algorithms**: Support for classification and regression
        - **Hyperparameter Tuning**: Optimize model performance
        - **Cross-Validation**: Robust model evaluation
        - **Feature Importance**: Understand model decisions
        - **Model Persistence**: Save and reuse trained models
        
        #### 🎨 Advanced Analytics
        - **Outlier Detection**: Identify anomalies in your data
        - **Distribution Analysis**: Understand data distributions
        - **Group Comparisons**: Compare statistics across categories
        - **Statistical Tests**: ANOVA, correlation tests, and more
        
        ### 🛠️ Technology Stack:
        - **Frontend**: Streamlit
        - **Backend**: FastAPI
        - **ML Framework**: Scikit-learn, XGBoost
        - **AI Agent**: LangChain + Google Gemini
        - **Visualization**: Matplotlib, Seaborn, Plotly
        
        ### 📝 How to Use:
        1. **Upload** your CSV dataset using the sidebar
        2. **Explore** your data with visualizations and statistics
        3. **Chat** with the AI to get insights
        4. **Train** machine learning models
        5. **Compare** different models and approaches
        
        ### 🎯 Best Practices:
        - Ensure your CSV file is properly formatted
        - Check for and handle missing values
        - Understand your data before training models
        - Use cross-validation for reliable metrics
        - Try multiple algorithms and compare results
        
        ### 🆘 Need Help?
        - Use the AI chat for data-related questions
        - Check the Help & Tips section in the sidebar
        - Explore the example datasets
        
        ### 📧 Contact & Support:
        For questions, suggestions, or bug reports, please reach out through:
        - GitHub Issues
        - Email: support@mlplayground.com
        
        ---
        
        Made with ❤️ by the ML Playground Team
        """)
        
        # Show some stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Version", "2.0.0")
        with col2:
            st.metric("Supported Models", "12+")
        with col3:
            st.metric("Analysis Tools", "15+")

if __name__ == "__main__":
    main()