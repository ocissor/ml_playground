# Required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import json

# Example function to train and evaluate models
def train_test_models(df, target_column, model_name: str, hyperparameter: str, problem_type='classification', test_size=0.2, random_state=42):
    """
    df: pandas DataFrame
    target_column: str, name of target column
    problem_type: 'classification' or 'regression'
    """

    models_list = [model_name]
    hyperparams = json.loads(hyperparameter)

    print(hyperparams)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    results = {}
    
    if problem_type == 'classification':
        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boost": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000)
        }
    else:  # regression
        models = {
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "SVM": SVR(),
            "AdaBoost": AdaBoostRegressor(),
            "Gradient Boost": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression()
        }
    

    for model_name in models_list:
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not recognized.")
        
        model_class = models.get(model_name).__class__
        model_instance = model_class(**hyperparams)
        print(model_instance)
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)
        
        if problem_type == 'classification':
            score = accuracy_score(y_test, y_pred)
            results[model_name] = {"accuracy": score}
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[model_name] = {"MSE": mse, "R2": r2}
    
    return results

