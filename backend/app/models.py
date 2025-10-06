import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")


def preprocess_data(df, target_column):
    """Preprocess the dataset for training"""
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.mean(numeric_only=True))
    
    # Encode target if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    return X, y


def get_model_params(model_name, hyperparams):
    """Extract relevant parameters for each model type"""
    params = {}
    
    if model_name == "logistic_regression":
        params = {
            'C': hyperparams.get('C', 1.0),
            'max_iter': int(hyperparams.get('max_iter', 1000)),
            'random_state': int(hyperparams.get('random_state', 42))
        }
    
    elif model_name == "random_forest":
        params = {
            'n_estimators': int(hyperparams.get('n_estimators', 100)),
            'max_depth': int(hyperparams.get('max_depth', 10)),
            'min_samples_split': int(hyperparams.get('min_samples_split', 2)),
            'random_state': int(hyperparams.get('random_state', 42))
        }
    
    elif model_name == "svm":
        params = {
            'C': float(hyperparams.get('C', 1.0)),
            'kernel': hyperparams.get('kernel', 'rbf'),
            'random_state': int(hyperparams.get('random_state', 42))
        }
        if params['kernel'] == 'rbf':
            params['gamma'] = hyperparams.get('gamma', 'scale')
    
    elif model_name == "gradient_boosting":
        params = {
            'n_estimators': int(hyperparams.get('n_estimators', 100)),
            'max_depth': int(hyperparams.get('max_depth', 3)),
            'learning_rate': float(hyperparams.get('learning_rate', 0.1)),
            'random_state': int(hyperparams.get('random_state', 42))
        }
    
    elif model_name == "xgboost":
        params = {
            'n_estimators': int(hyperparams.get('n_estimators', 100)),
            'max_depth': int(hyperparams.get('max_depth', 6)),
            'learning_rate': float(hyperparams.get('learning_rate', 0.1)),
            'random_state': int(hyperparams.get('random_state', 42))
        }
    
    elif model_name == "knn":
        params = {
            'n_neighbors': int(hyperparams.get('n_neighbors', 5)),
            'weights': hyperparams.get('weights', 'uniform')
        }
    
    elif model_name == "decision_tree":
        params = {
            'max_depth': int(hyperparams.get('max_depth', 10)),
            'min_samples_split': int(hyperparams.get('min_samples_split', 2)),
            'random_state': int(hyperparams.get('random_state', 42))
        }
    
    elif model_name == "linear_regression":
        params = {}
    
    elif model_name == "ridge":
        params = {
            'alpha': float(hyperparams.get('alpha', 1.0)),
            'random_state': int(hyperparams.get('random_state', 42))
        }
    
    elif model_name == "lasso":
        params = {
            'alpha': float(hyperparams.get('alpha', 1.0)),
            'random_state': int(hyperparams.get('random_state', 42))
        }
    
    elif model_name == "svr":
        params = {
            'C': float(hyperparams.get('C', 1.0)),
            'kernel': hyperparams.get('kernel', 'rbf')
        }
        if params['kernel'] == 'rbf':
            params['gamma'] = hyperparams.get('gamma', 'scale')
    
    return params


def get_model(model_name, problem_type, hyperparams):
    """Initialize the appropriate model with parameters"""
    params = get_model_params(model_name, hyperparams)
    
    # Classification models
    if problem_type == "classification":
        if model_name == "logistic_regression":
            return LogisticRegression(**params)
        elif model_name == "random_forest":
            return RandomForestClassifier(**params)
        elif model_name == "svm":
            return SVC(**params)
        elif model_name == "gradient_boosting":
            return GradientBoostingClassifier(**params)
        elif model_name == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is not installed")
            return XGBClassifier(**params, eval_metric='logloss')
        elif model_name == "knn":
            return KNeighborsClassifier(**params)
        elif model_name == "decision_tree":
            return DecisionTreeClassifier(**params)
    
    # Regression models
    elif problem_type == "regression":
        if model_name == "linear_regression":
            return LinearRegression(**params)
        elif model_name == "ridge":
            return Ridge(**params)
        elif model_name == "lasso":
            return Lasso(**params)
        elif model_name == "random_forest":
            return RandomForestRegressor(**params)
        elif model_name == "gradient_boosting":
            return GradientBoostingRegressor(**params)
        elif model_name == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is not installed")
            return XGBRegressor(**params)
        elif model_name == "svr":
            return SVR(**params)
        elif model_name == "decision_tree":
            return DecisionTreeRegressor(**params)
    
    raise ValueError(f"Unknown model: {model_name} for problem type: {problem_type}")


def auto_detect_problem_type(y, specified_type=None):
    """
    Automatically detect if the problem is classification or regression
    
    Args:
        y: Target variable
        specified_type: User-specified problem type (optional)
    
    Returns:
        'classification' or 'regression'
    """
    # If user specified, validate it
    if specified_type:
        # Check if it makes sense
        n_unique = len(np.unique(y))
        
        if specified_type == "classification":
            # Check if target has too many unique values for classification
            if n_unique > 50 and y.dtype in ['float64', 'float32']:
                raise ValueError(
                    f"Target has {n_unique} unique continuous values. "
                    f"This appears to be a regression problem. "
                    f"Please select 'regression' as the problem type."
                )
            return "classification"
        
        elif specified_type == "regression":
            # Check if target is discrete (likely classification)
            if n_unique <= 20 and y.dtype in ['int64', 'int32', 'object']:
                raise ValueError(
                    f"Target has only {n_unique} unique discrete values. "
                    f"This appears to be a classification problem. "
                    f"Please select 'classification' as the problem type."
                )
            return "regression"
    
    # Auto-detect
    n_unique = len(np.unique(y))
    
    # If target is object/string, it's classification
    if y.dtype == 'object':
        return "classification"
    
    # If less than 20 unique values, likely classification
    if n_unique < 20:
        return "classification"
    
    # If many unique values and continuous, it's regression
    if n_unique > 50:
        return "regression"
    
    # Edge case: between 20-50 unique values
    # Check if values look continuous
    if y.dtype in ['float64', 'float32']:
        return "regression"
    else:
        return "classification"


def train_test_models(df, target_column, model_name, hyperparameter, problem_type="classification"):
    """
    Train and evaluate machine learning models
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        model_name: Name of the model to train
        hyperparameter: JSON string of hyperparameters
        problem_type: 'classification' or 'regression'
    
    Returns:
        Dictionary containing model metrics and results
    """
    try:
        # Parse hyperparameters
        try:
            hyperparams = json.loads(hyperparameter)
        except json.JSONDecodeError:
            hyperparams = {}
        
        # Extract test_size and cv_folds separately
        test_size = float(hyperparams.pop('test_size', 0.2))
        cv_folds = int(hyperparams.pop('cv_folds', 5))
        random_state = int(hyperparams.get('random_state', 42))
        
        # Preprocess data
        X, y = preprocess_data(df, target_column)
        
        # Auto-detect and validate problem type
        detected_type = auto_detect_problem_type(y, problem_type)
        
        if detected_type != problem_type:
            raise ValueError(
                f"Problem type mismatch! You selected '{problem_type}' but the target variable "
                f"appears to be for '{detected_type}'. "
                f"Target has {len(np.unique(y))} unique values. "
                f"Please select the correct problem type."
            )
        
        problem_type = detected_type
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features for certain models
        if model_name in ['svm', 'svr', 'knn', 'logistic_regression']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Initialize and train model
        model = get_model(model_name, problem_type, hyperparams)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Prepare results
        results = {
            'model_name': model_name,
            'problem_type': problem_type,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        # Calculate metrics based on problem type
        if problem_type == "classification":
            results['metrics'] = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            results['classification_report'] = report
            
            # Cross-validation score
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=min(cv_folds, 5), scoring='accuracy')
                results['cv_scores'] = {
                    'mean': float(cv_scores.mean()),
                    'std': float(cv_scores.std()),
                    'scores': cv_scores.tolist()
                }
            except Exception as e:
                print(f"CV failed: {e}")
        
        else:  # regression
            mse = mean_squared_error(y_test, y_pred)
            results['metrics'] = {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2_score': float(r2_score(y_test, y_pred))
            }
            
            # Store predictions for plotting
            results['predictions'] = y_pred.tolist()[:100]  # Limit to 100 for performance
            results['actuals'] = y_test.tolist()[:100]
            
            # Cross-validation score
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=min(cv_folds, 5), scoring='r2')
                results['cv_scores'] = {
                    'mean': float(cv_scores.mean()),
                    'std': float(cv_scores.std()),
                    'scores': cv_scores.tolist()
                }
            except Exception as e:
                print(f"CV failed: {e}")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = [
                {'feature': col, 'importance': float(imp)}
                for col, imp in zip(X.columns, importances)
            ]
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            results['feature_importance'] = feature_importance
        elif hasattr(model, 'coef_'):
            # For linear models
            coefs = model.coef_
            if len(coefs.shape) > 1:
                coefs = np.abs(coefs).mean(axis=0)
            feature_importance = [
                {'feature': col, 'importance': float(abs(coef))}
                for col, coef in zip(X.columns, coefs)
            ]
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            results['feature_importance'] = feature_importance
        
        return results
    
    except Exception as e:
        raise Exception(f"Training failed: {str(e)}")


# Additional utility functions
def get_available_models(problem_type):
    """Get list of available models for a problem type"""
    if problem_type == "classification":
        models = {
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest",
            "svm": "Support Vector Machine",
            "gradient_boosting": "Gradient Boosting",
            "knn": "K-Nearest Neighbors",
            "decision_tree": "Decision Tree"
        }
        if XGBOOST_AVAILABLE:
            models["xgboost"] = "XGBoost"
    else:
        models = {
            "linear_regression": "Linear Regression",
            "ridge": "Ridge Regression",
            "lasso": "Lasso Regression",
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting",
            "svr": "Support Vector Regression",
            "decision_tree": "Decision Tree"
        }
        if XGBOOST_AVAILABLE:
            models["xgboost"] = "XGBoost"
    
    return models


def validate_hyperparameters(model_name, hyperparams):
    """Validate hyperparameters for a given model"""
    # This can be expanded with specific validation rules
    validated = {}
    
    for key, value in hyperparams.items():
        if key in ['n_estimators', 'max_depth', 'min_samples_split', 'n_neighbors', 'max_iter', 'cv_folds']:
            validated[key] = int(value)
        elif key in ['C', 'alpha', 'learning_rate', 'test_size']:
            validated[key] = float(value)
        elif key == 'random_state':
            validated[key] = int(value)
        else:
            validated[key] = value
    
    return validated