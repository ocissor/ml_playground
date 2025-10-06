from fastapi import FastAPI, UploadFile, Form, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from backend.app.models import train_test_models
from typing import List, Optional
from pathlib import Path
import sys
import json
sys.path.append(Path(__file__).parent)
from backend.app.eda_plot import *
from backend.app.graph.graph import analyze_data_graph
from backend.app.data_storage import dataframe, conversation_history
import pickle
import numpy as np
from scipy import stats

app = FastAPI(
    title="ML Playground Backend",
    description="Advanced ML and Data Analytics API",
    version="2.0.0"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== UTILITY FUNCTIONS ============== #

def validate_dataframe(df: pd.DataFrame) -> dict:
    """Validate and return basic info about dataframe"""
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file contains no data")
    
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_usage": df.memory_usage(deep=True).sum(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

def detect_outliers(df: pd.DataFrame, column: str) -> dict:
    """Detect outliers using IQR method"""
    if column not in df.columns:
        return {"error": f"Column {column} not found"}
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return {
        "total_outliers": len(outliers),
        "percentage": (len(outliers) / len(df)) * 100,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_indices": outliers.index.tolist()[:100]  # Limit to 100
    }

# ============== EXISTING ENDPOINTS (Enhanced) ============== #

@app.post('/detect_problem_type')
async def detect_problem_type(file: UploadFile, target_column: str = Form(...)):
    """Automatically detect if the problem is classification or regression"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        y = df[target_column]
        
        # Handle categorical
        if y.dtype == 'object':
            y_numeric = pd.factorize(y)[0]
        else:
            y_numeric = y
        
        n_unique = len(np.unique(y_numeric))
        
        # Detection logic
        if y.dtype == 'object':
            problem_type = "classification"
            reason = f"Target is categorical (text values)"
        elif n_unique < 20:
            problem_type = "classification"
            reason = f"Target has {n_unique} unique values (likely discrete classes)"
        elif n_unique > 50:
            problem_type = "regression"
            reason = f"Target has {n_unique} unique continuous values"
        elif y.dtype in ['float64', 'float32']:
            problem_type = "regression"
            reason = f"Target has continuous float values ({n_unique} unique)"
        else:
            problem_type = "classification"
            reason = f"Target has {n_unique} unique integer values"
        
        return {
            "problem_type": problem_type,
            "reason": reason,
            "unique_values": int(n_unique),
            "target_dtype": str(y.dtype),
            "sample_values": y.head(10).tolist(),
            "is_continuous": bool(y.dtype in ['float64', 'float32'] and n_unique > 50)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/train")
async def train_model(
    file: UploadFile, 
    target_column: str = Form(...), 
    model_name: str = Form(...), 
    problem_type: str = Form("classification"), 
    hyperparameter: str = Form("{}")
):
    """Train ML model with enhanced error handling"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate
        validate_dataframe(df)
        
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        # Validate hyperparameters
        try:
            hyperparams = json.loads(hyperparameter) if hyperparameter else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid hyperparameter JSON format")
        
        result = train_test_models(df, target_column, model_name, json.dumps(hyperparams), problem_type=problem_type)
        return {'model_outputs': result, 'status': 'success'}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/data_stats")
async def get_data_stats(file: UploadFile):
    """Get comprehensive data statistics"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        validate_dataframe(df)
        
        # Basic statistics
        stats_dict = {
            'basic_stats': df.describe().to_json(),
            'info': {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'missing_values': df.isnull().sum().to_dict(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            }
        }
        
        return stats_dict
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics generation failed: {str(e)}")

@app.post('/visualize_data')
async def visualize_data(file: UploadFile):
    """Generate visualizations with error handling"""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        df = pd.read_csv(io.BytesIO(contents))
        validate_dataframe(df)
        
        images = univariate_plots(df)
        
        if not images:
            return {"message": "No visualizations could be generated", "images": {}}
        
        return images
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@app.post('/chat_with_data')
async def chat_with_data(file: UploadFile, user_input: str = Form(...), uuid: str = Form(...)):
    """Enhanced chat with data endpoint"""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        df = pd.read_csv(io.BytesIO(contents))
        validate_dataframe(df)
        
        # Initialize conversation history
        if uuid not in conversation_history:
            conversation_history[uuid] = []
        
        graph_input = {
            'user_input': user_input,
            'messages': conversation_history[uuid],
            'uuid': uuid
        }
        
        output = analyze_data_graph.invoke(graph_input, config={"configurable": {"thread_id": uuid}})
        conversation_history[uuid] = output['messages']
        
        return Response(content=pickle.dumps(output), media_type="application/octet-stream")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post('/upload_file')
async def save_data_to_dataframe(file: UploadFile, uuid: str = Form(...)):
    """Save uploaded file to dataframe storage"""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        df = pd.read_csv(io.BytesIO(content))

        dataframe[uuid] = df.to_json(orient='records')  # or orient='split', 'columns' etc.

        validate_dataframe(df)
        
        if uuid in dataframe:
            return {"message": "UUID already exists. Data not overwritten.", "status": "warning"}
        
        dataframe[uuid] = df.to_json(orient="records")
        
        return {
            "message": "File uploaded and data saved successfully.",
            "status": "success",
            "data_info": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": df.columns.tolist()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ============== NEW ENDPOINTS ============== #

@app.post('/data_profiling')
async def data_profiling(file: UploadFile):
    """Generate comprehensive data profiling report"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        validate_dataframe(df)
        
        profile = {
            "overview": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
                "missing_cells": df.isnull().sum().sum(),
                "missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
                "duplicate_rows": df.duplicated().sum()
            },
            "columns": {}
        }
        
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isnull().sum()),
                "missing_percentage": float((df[col].isnull().sum() / len(df)) * 100),
                "unique": int(df[col].nunique()),
                "unique_percentage": float((df[col].nunique() / len(df)) * 100)
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                    "median": float(df[col].median()) if not df[col].isnull().all() else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "zeros": int((df[col] == 0).sum()),
                    "zeros_percentage": float(((df[col] == 0).sum() / len(df)) * 100)
                })
                
                # Outlier detection
                outlier_info = detect_outliers(df, col)
                col_info["outliers"] = outlier_info
            
            else:
                # For categorical columns
                top_values = df[col].value_counts().head(10).to_dict()
                col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
            
            profile["columns"][col] = col_info
        
        return profile
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")

@app.post('/correlation_analysis')
async def correlation_analysis(file: UploadFile, method: str = Form("pearson")):
    """Calculate correlation matrix with different methods"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        validate_dataframe(df)
        
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 numeric columns for correlation")
        
        if method not in ['pearson', 'spearman', 'kendall']:
            method = 'pearson'
        
        corr_matrix = numeric_df.corr(method=method)
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_val),
                        "strength": "strong" if abs(corr_val) > 0.7 else "moderate"
                    })
        
        # Sort by absolute correlation
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            "correlation_matrix": corr_matrix.to_json(),
            "method": method,
            "strong_correlations": strong_correlations,
            "summary": {
                "total_features": len(corr_matrix.columns),
                "strong_correlations_count": len([c for c in strong_correlations if c['strength'] == 'strong']),
                "moderate_correlations_count": len([c for c in strong_correlations if c['strength'] == 'moderate'])
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correlation analysis failed: {str(e)}")

@app.post('/missing_data_analysis')
async def missing_data_analysis(file: UploadFile):
    """Analyze missing data patterns"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        validate_dataframe(df)
        
        missing_summary = []
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_summary.append({
                    "column": col,
                    "missing_count": int(missing_count),
                    "missing_percentage": float((missing_count / len(df)) * 100),
                    "dtype": str(df[col].dtype)
                })
        
        # Sort by missing percentage
        missing_summary.sort(key=lambda x: x['missing_percentage'], reverse=True)
        
        return {
            "total_missing_cells": int(total_missing),
            "total_cells": int(total_cells),
            "overall_missing_percentage": float((total_missing / total_cells) * 100),
            "columns_with_missing": missing_summary,
            "complete_rows": int(df.dropna().shape[0]),
            "complete_rows_percentage": float((df.dropna().shape[0] / len(df)) * 100)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Missing data analysis failed: {str(e)}")

@app.post('/outlier_detection')
async def outlier_detection_endpoint(file: UploadFile, columns: Optional[str] = Form(None)):
    """Detect outliers in numeric columns"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        validate_dataframe(df)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if columns:
            target_cols = [c.strip() for c in columns.split(',')]
            target_cols = [c for c in target_cols if c in numeric_cols]
        else:
            target_cols = numeric_cols
        
        outlier_report = {}
        
        for col in target_cols:
            outlier_info = detect_outliers(df, col)
            outlier_report[col] = outlier_info
        
        return {
            "outlier_analysis": outlier_report,
            "summary": {
                "total_columns_analyzed": len(target_cols),
                "columns_with_outliers": len([c for c, info in outlier_report.items() 
                                             if 'total_outliers' in info and info['total_outliers'] > 0])
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outlier detection failed: {str(e)}")

@app.post('/distribution_analysis')
async def distribution_analysis(file: UploadFile, column: str = Form(...)):
    """Analyze distribution of a column"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        validate_dataframe(df)
        
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found")
        
        col_data = df[column].dropna()
        
        if pd.api.types.is_numeric_dtype(col_data):
            # Normality test
            if len(col_data) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(col_data[:5000])  # Limit for performance
            else:
                shapiro_stat, shapiro_p = None, None
            
            analysis = {
                "column": column,
                "type": "numeric",
                "statistics": {
                    "count": int(len(col_data)),
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "mode": float(col_data.mode()[0]) if len(col_data.mode()) > 0 else None,
                    "std": float(col_data.std()),
                    "variance": float(col_data.var()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "range": float(col_data.max() - col_data.min()),
                    "q1": float(col_data.quantile(0.25)),
                    "q2": float(col_data.quantile(0.50)),
                    "q3": float(col_data.quantile(0.75)),
                    "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                    "skewness": float(col_data.skew()),
                    "kurtosis": float(col_data.kurtosis())
                },
                "normality_test": {
                    "test": "Shapiro-Wilk",
                    "statistic": float(shapiro_stat) if shapiro_stat is not None else None,
                    "p_value": float(shapiro_p) if shapiro_p is not None else None,
                    "is_normal": bool(shapiro_p > 0.05) if shapiro_p is not None else None
                }
            }
        else:
            # Categorical analysis
            value_counts = col_data.value_counts()
            analysis = {
                "column": column,
                "type": "categorical",
                "statistics": {
                    "count": int(len(col_data)),
                    "unique": int(col_data.nunique()),
                    "top_value": str(value_counts.index[0]),
                    "top_frequency": int(value_counts.iloc[0]),
                    "top_percentage": float((value_counts.iloc[0] / len(col_data)) * 100)
                },
                "value_counts": {str(k): int(v) for k, v in value_counts.head(20).items()}
            }
        
        return analysis
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Distribution analysis failed: {str(e)}")

@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "active_sessions": len(dataframe),
        "active_conversations": len(conversation_history)
    }

@app.delete('/clear_session/{uuid}')
async def clear_session(uuid: str):
    """Clear session data for a specific UUID"""
    try:
        removed_data = uuid in dataframe
        removed_history = uuid in conversation_history
        
        if uuid in dataframe:
            del dataframe[uuid]
        if uuid in conversation_history:
            del conversation_history[uuid]
        
        return {
            "message": "Session cleared successfully",
            "removed_data": removed_data,
            "removed_history": removed_history
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session cleanup failed: {str(e)}")

@app.get('/list_sessions')
async def list_sessions():
    """List all active sessions"""
    return {
        "active_data_sessions": list(dataframe.keys()),
        "active_chat_sessions": list(conversation_history.keys()),
        "total_data_sessions": len(dataframe),
        "total_chat_sessions": len(conversation_history)
    }