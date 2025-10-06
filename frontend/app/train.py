import streamlit as st
import requests
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

def train_models(app_state):
    """Model training interface with advanced features"""
    
    st.title("🤖 Machine Learning Model Training")
    st.markdown("### Train and evaluate machine learning models on your dataset")
    
    # Load dataset info
    try:
        app_state.uploaded_file.seek(0)
        df = pd.read_csv(app_state.uploaded_file)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["⚙️ Configure & Train", "📊 Results & Metrics", "💾 Model Management"])
    
    # ==================== TAB 1: CONFIGURE & TRAIN ==================== #
    with tab1:
        st.subheader("⚙️ Model Configuration")
        
        # Problem Type Selection
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Initialize problem_type
            problem_type = "classification"
        
        with col2:
            # Target column selection
            target_column = st.selectbox(
                "Select Target Column",
                df.columns.tolist(),
                help="The column you want to predict"
            )
        
        # Auto-detect problem type when target changes
        if target_column:
            with st.spinner("Analyzing target variable..."):
                try:
                    app_state.uploaded_file.seek(0)
                    response = requests.post(
                        "http://ml-playground-backend-service:8000/detect_problem_type",
                        files={"file": app_state.uploaded_file},
                        data={"target_column": target_column}
                    )
                    
                    if response.status_code == 200:
                        detection = response.json()
                        detected_type = detection['problem_type']
                        
                        # Show detection result in col1
                        with col1:
                            st.markdown("#### 🎯 Problem Type")
                            st.success(f"**{detected_type.capitalize()}**")
                            st.caption(f"✓ {detection['reason']}")
                        
                        # Show metrics
                        st.markdown("---")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Unique Values", detection['unique_values'])
                        with col_b:
                            st.metric("Data Type", detection['target_dtype'])
                        with col_c:
                            continuous = "Yes" if detection['is_continuous'] else "No"
                            st.metric("Continuous", continuous)
                        
                        # Override problem_type with detection
                        problem_type = detected_type
                        
                        # Show manual override option
                        with st.expander("⚙️ Override Problem Type (Advanced)"):
                            manual_type = st.radio(
                                "Force problem type:",
                                ["classification", "regression"],
                                index=0 if detected_type == "classification" else 1
                            )
                            if manual_type != detected_type:
                                st.warning(f"⚠️ You're overriding auto-detection. Make sure this is correct!")
                                problem_type = manual_type
                    
                except Exception as e:
                    st.warning(f"Could not auto-detect problem type. Please select manually.")
                    with col1:
                        problem_type = st.selectbox(
                            "Select Problem Type",
                            ["classification", "regression"],
                            help="Choose the type of machine learning problem"
                        )
        else:
            with col1:
                st.info("Select a target column to auto-detect problem type")
        
        # Show target distribution
        if target_column:
            st.markdown("#### 🎯 Target Variable Distribution")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if problem_type == "classification":
                    value_counts = df[target_column].value_counts()
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        labels={'x': target_column, 'y': 'Count'},
                        title=f"Distribution of {target_column}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(
                        df,
                        x=target_column,
                        title=f"Distribution of {target_column}",
                        nbins=30
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("##### 📊 Statistics")
                if problem_type == "classification":
                    st.write(f"**Unique Values:** {df[target_column].nunique()}")
                    st.write(f"**Most Common:** {df[target_column].mode()[0]}")
                    st.write(f"**Class Balance:**")
                    for val, count in df[target_column].value_counts().items():
                        pct = (count / len(df)) * 100
                        st.write(f"  {val}: {pct:.1f}%")
                else:
                    st.write(f"**Mean:** {df[target_column].mean():.4f}")
                    st.write(f"**Median:** {df[target_column].median():.4f}")
                    st.write(f"**Std Dev:** {df[target_column].std():.4f}")
                    st.write(f"**Min:** {df[target_column].min():.4f}")
                    st.write(f"**Max:** {df[target_column].max():.4f}")
        
        st.markdown("---")
        
        # Model Selection
        st.markdown("#### 🎯 Select Algorithm")
        
        if problem_type == "classification":
            model_options = {
                "Logistic Regression": "logistic_regression",
                "Random Forest": "random_forest",
                "Support Vector Machine": "svm",
                "Gradient Boosting": "gradient_boosting",
                "XGBoost": "xgboost",
                "K-Nearest Neighbors": "knn",
                "Decision Tree": "decision_tree"
            }
        else:
            model_options = {
                "Linear Regression": "linear_regression",
                "Ridge Regression": "ridge",
                "Lasso Regression": "lasso",
                "Random Forest": "random_forest",
                "Gradient Boosting": "gradient_boosting",
                "XGBoost": "xgboost",
                "Support Vector Regression": "svr"
            }
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_model_name = st.selectbox(
                "Choose Model",
                list(model_options.keys()),
                help="Select the machine learning algorithm to train"
            )
            model_name = model_options[selected_model_name]
        
        with col2:
            st.markdown("##### ℹ️ Model Info")
            model_info = {
                "logistic_regression": "Fast, interpretable binary/multiclass classifier",
                "random_forest": "Ensemble method, handles non-linear relationships",
                "svm": "Effective in high-dimensional spaces",
                "gradient_boosting": "Sequential ensemble, high accuracy",
                "xgboost": "Optimized gradient boosting, very fast",
                "knn": "Instance-based learning, simple but effective",
                "decision_tree": "Interpretable, prone to overfitting",
                "linear_regression": "Simple, fast, assumes linear relationship",
                "ridge": "Linear regression with L2 regularization",
                "lasso": "Linear regression with L1 regularization",
                "svr": "SVM for regression tasks"
            }
            st.info(model_info.get(model_name, "No description available"))
        
        st.markdown("---")
        
        # Hyperparameter Configuration
        st.markdown("#### ⚙️ Hyperparameters")
        
        hyperparams = {}
        
        with st.expander("Configure Hyperparameters", expanded=True):
            if model_name == "logistic_regression":
                col1, col2 = st.columns(2)
                with col1:
                    hyperparams['C'] = st.slider("Regularization (C)", 0.001, 10.0, 1.0, 0.001)
                with col2:
                    hyperparams['max_iter'] = st.number_input("Max Iterations", 100, 10000, 1000)
            
            elif model_name in ["random_forest", "gradient_boosting", "xgboost"]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    hyperparams['n_estimators'] = st.slider("Number of Trees", 10, 500, 100)
                with col2:
                    hyperparams['max_depth'] = st.slider("Max Depth", 1, 50, 10)
                with col3:
                    hyperparams['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2)
                
                if model_name == "xgboost":
                    hyperparams['learning_rate'] = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
            
            elif model_name == "svm" or model_name == "svr":
                col1, col2, col3 = st.columns(3)
                with col1:
                    hyperparams['C'] = st.slider("Regularization (C)", 0.1, 100.0, 1.0, 0.1)
                with col2:
                    hyperparams['kernel'] = st.selectbox("Kernel", ["rbf", "linear", "poly"])
                with col3:
                    if hyperparams['kernel'] == 'rbf':
                        hyperparams['gamma'] = st.selectbox("Gamma", ["scale", "auto"])
            
            elif model_name == "knn":
                col1, col2 = st.columns(2)
                with col1:
                    hyperparams['n_neighbors'] = st.slider("Number of Neighbors", 1, 50, 5)
                with col2:
                    hyperparams['weights'] = st.selectbox("Weights", ["uniform", "distance"])
            
            elif model_name == "decision_tree":
                col1, col2 = st.columns(2)
                with col1:
                    hyperparams['max_depth'] = st.slider("Max Depth", 1, 50, 10)
                with col2:
                    hyperparams['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2)
            
            elif model_name in ["ridge", "lasso"]:
                hyperparams['alpha'] = st.slider("Regularization Alpha", 0.001, 10.0, 1.0, 0.001)
        
        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
                hyperparams['test_size'] = test_size
            with col2:
                random_state = st.number_input("Random State (for reproducibility)", 0, 1000, 42)
                hyperparams['random_state'] = random_state
            
            use_cv = st.checkbox("Use Cross-Validation", value=True)
            if use_cv:
                cv_folds = st.slider("Number of CV Folds", 2, 10, 5)
                hyperparams['cv_folds'] = cv_folds
        
        # Convert hyperparams to JSON string
        hyperparameter_json = json.dumps(hyperparams)
        
        st.markdown("---")
        
        # Train Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)
        
        if train_button:
            if not target_column:
                st.error("Please select a target column!")
            else:
                with st.spinner(f"Training {selected_model_name}... This may take a moment."):
                    try:
                        app_state.uploaded_file.seek(0)
                        response = requests.post(
                            "http://ml-playground-backend-service:8000/train",
                            files={"file": app_state.uploaded_file},
                            data={
                                "target_column": target_column,
                                "model_name": model_name,
                                "problem_type": problem_type,
                                "hyperparameter": hyperparameter_json
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Store results in session state
                            if 'training_results' not in st.session_state:
                                st.session_state.training_results = []
                            
                            st.session_state.training_results.append({
                                'model_name': selected_model_name,
                                'problem_type': problem_type,
                                'target': target_column,
                                'results': result['model_outputs'],
                                'hyperparams': hyperparams
                            })
                            
                            st.success("✅ Model trained successfully!")
                            st.balloons()
                            
                        else:
                            st.error(f"Training failed: {response.json().get('detail', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
    
    # ==================== TAB 2: RESULTS & METRICS ==================== #
    with tab2:
        st.subheader("📊 Training Results & Performance Metrics")
        
        if 'training_results' not in st.session_state or not st.session_state.training_results:
            st.info("No trained models yet. Train a model in the 'Configure & Train' tab to see results here.")
        else:
            # Model selector
            model_names = [f"{r['model_name']} (Target: {r['target']})" for r in st.session_state.training_results]
            selected_result_idx = st.selectbox(
                "Select Model to View",
                range(len(model_names)),
                format_func=lambda x: model_names[x]
            )
            
            result_data = st.session_state.training_results[selected_result_idx]
            results = result_data['results']
            
            st.markdown(f"### {result_data['model_name']}")
            st.markdown(f"**Problem Type:** {result_data['problem_type'].capitalize()}")
            st.markdown(f"**Target Variable:** {result_data['target']}")
            
            st.markdown("---")
            
            # Display metrics based on problem type
            if result_data['problem_type'] == 'classification':
                # Classification Metrics
                st.markdown("#### 📈 Classification Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                metrics = results.get('metrics', {})
                
                with col1:
                    accuracy = metrics.get('accuracy', 0)
                    st.metric("Accuracy", f"{accuracy:.4f}", delta=f"{(accuracy - 0.5) * 100:.1f}%")
                
                with col2:
                    precision = metrics.get('precision', 0)
                    st.metric("Precision", f"{precision:.4f}")
                
                with col3:
                    recall = metrics.get('recall', 0)
                    st.metric("Recall", f"{recall:.4f}")
                
                with col4:
                    f1 = metrics.get('f1_score', 0)
                    st.metric("F1-Score", f"{f1:.4f}")
                
                # Confusion Matrix (if available)
                if 'confusion_matrix' in results:
                    st.markdown("#### 🎯 Confusion Matrix")
                    cm = results['confusion_matrix']
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=[f"Pred {i}" for i in range(len(cm))],
                        y=[f"True {i}" for i in range(len(cm))],
                        colorscale='Blues',
                        text=cm,
                        texttemplate="%{text}",
                        textfont={"size": 16}
                    ))
                    fig.update_layout(
                        title="Confusion Matrix",
                        xaxis_title="Predicted Label",
                        yaxis_title="True Label",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Classification Report
                if 'classification_report' in results:
                    st.markdown("#### 📋 Detailed Classification Report")
                    report_df = pd.DataFrame(results['classification_report']).T
                    st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
            
            else:
                # Regression Metrics
                st.markdown("#### 📈 Regression Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                metrics = results.get('metrics', {})
                
                with col1:
                    mse = metrics.get('mse', 0)
                    st.metric("MSE", f"{mse:.4f}")
                
                with col2:
                    rmse = metrics.get('rmse', 0)
                    st.metric("RMSE", f"{rmse:.4f}")
                
                with col3:
                    mae = metrics.get('mae', 0)
                    st.metric("MAE", f"{mae:.4f}")
                
                with col4:
                    r2 = metrics.get('r2_score', 0)
                    st.metric("R² Score", f"{r2:.4f}", delta=f"{r2 * 100:.1f}%")
                
                # Predictions vs Actual Plot
                if 'predictions' in results and 'actuals' in results:
                    st.markdown("#### 📊 Predictions vs Actual Values")
                    
                    fig = go.Figure()
                    
                    # Scatter plot
                    fig.add_trace(go.Scatter(
                        x=results['actuals'],
                        y=results['predictions'],
                        mode='markers',
                        name='Predictions',
                        marker=dict(size=8, opacity=0.6)
                    ))
                    
                    # Perfect prediction line
                    min_val = min(min(results['actuals']), min(results['predictions']))
                    max_val = max(max(results['actuals']), max(results['predictions']))
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Predicted vs Actual Values",
                        xaxis_title="Actual Values",
                        yaxis_title="Predicted Values",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Residuals Plot
                if 'predictions' in results and 'actuals' in results:
                    st.markdown("#### 📉 Residuals Analysis")
                    
                    residuals = [a - p for a, p in zip(results['actuals'], results['predictions'])]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=results['predictions'],
                            y=residuals,
                            mode='markers',
                            marker=dict(size=8, opacity=0.6)
                        ))
                        fig.add_hline(y=0, line_dash="dash", line_color="red")
                        fig.update_layout(
                            title="Residual Plot",
                            xaxis_title="Predicted Values",
                            yaxis_title="Residuals",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=residuals,
                            nbinsx=30,
                            name='Residuals'
                        ))
                        fig.update_layout(
                            title="Distribution of Residuals",
                            xaxis_title="Residual Value",
                            yaxis_title="Frequency",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            if 'feature_importance' in results:
                st.markdown("---")
                st.markdown("#### 🎯 Feature Importance")
                
                importance_df = pd.DataFrame(results['feature_importance'])
                importance_df = importance_df.sort_values('importance', ascending=False).head(15)
                
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 15 Most Important Features'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Hyperparameters Used
            st.markdown("---")
            st.markdown("#### ⚙️ Hyperparameters Used")
            hyperparams_df = pd.DataFrame(result_data['hyperparams'].items(), columns=['Parameter', 'Value'])
            st.dataframe(hyperparams_df, use_container_width=True)
    
    # ==================== TAB 3: MODEL MANAGEMENT ==================== #
    with tab3:
        st.subheader("💾 Model Management")
        
        if 'training_results' not in st.session_state or not st.session_state.training_results:
            st.info("No trained models available. Train some models first!")
        else:
            st.markdown("#### 📚 Trained Models")
            
            # Create a summary table
            model_summary = []
            for idx, result in enumerate(st.session_state.training_results):
                metrics = result['results'].get('metrics', {})
                
                if result['problem_type'] == 'classification':
                    primary_metric = metrics.get('accuracy', 0)
                    metric_name = 'Accuracy'
                else:
                    primary_metric = metrics.get('r2_score', 0)
                    metric_name = 'R² Score'
                
                model_summary.append({
                    'Index': idx,
                    'Model': result['model_name'],
                    'Type': result['problem_type'].capitalize(),
                    'Target': result['target'],
                    metric_name: f"{primary_metric:.4f}"
                })
            
            summary_df = pd.DataFrame(model_summary)
            st.dataframe(summary_df, use_container_width=True)
            
            st.markdown("---")
            
            # Model Actions
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🗑️ Clear Models")
                if st.button("Clear All Trained Models", type="secondary"):
                    st.session_state.training_results = []
                    st.success("All models cleared!")
                    st.rerun()
            
            with col2:
                st.markdown("#### 📥 Export Results")
                if st.button("Download Model Summary", type="secondary"):
                    summary_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=summary_csv,
                        file_name="model_summary.csv",
                        mime="text/csv"
                    )
            
            st.markdown("---")
            
            # Model Comparison
            if len(st.session_state.training_results) > 1:
                st.markdown("#### 📊 Model Comparison")
                
                comparison_data = []
                for result in st.session_state.training_results:
                    metrics = result['results'].get('metrics', {})
                    row = {
                        'Model': result['model_name'],
                        'Type': result['problem_type']
                    }
                    row.update(metrics)
                    comparison_data.append(row)
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.style.highlight_max(axis=0), use_container_width=True)
                
                # Visualization
                if result['problem_type'] == 'classification':
                    metric_cols = ['accuracy', 'precision', 'recall', 'f1_score']
                else:
                    metric_cols = ['r2_score', 'mse', 'mae', 'rmse']
                
                available_metrics = [col for col in metric_cols if col in comparison_df.columns]
                
                if available_metrics:
                    fig = go.Figure()
                    
                    for metric in available_metrics:
                        fig.add_trace(go.Bar(
                            name=metric.upper(),
                            x=comparison_df['Model'],
                            y=comparison_df[metric]
                        ))
                    
                    fig.update_layout(
                        title="Model Performance Comparison",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)