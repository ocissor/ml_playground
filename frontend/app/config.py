from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

model_hyperparams = {
    "Decision Tree": DecisionTreeClassifier().get_params(),
    "Random Forest": RandomForestClassifier().get_params(),
    "SVM": SVC().get_params(),
    "AdaBoost": AdaBoostClassifier().get_params(),
    "Gradient Boost": GradientBoostingClassifier().get_params(),
    "Logistic Regression": LogisticRegression().get_params(),
}

model_names = [
    "Decision Tree",
    "Random Forest",
    "SVM",
    "AdaBoost",
    "Gradient Boost",
    "Logistic Regression"
]

model_hyperparams_values = {
    "Decision Tree": {
        "criterion": {"default": "gini", "options": ["gini", "entropy", "log_loss"]},
        "splitter": {"default": "best", "options": ["best", "random"]},
        "max_depth": {"default": None, "options": [None] + list(range(1, 51))},
        "min_samples_split": {"default": 2, "options": list(range(2, 21))},  # only ints
        "min_samples_leaf": {"default": 1, "options": list(range(1, 21))},  # only ints
        "min_weight_fraction_leaf": {"default": 0.0, "options": [i/100 for i in range(0, 51)]},
        "max_features": {"default": None, "options": ["sqrt", "log2", None]},
        "random_state": {"default": None, "options": [None] + list(range(0, 101))},
        "max_leaf_nodes": {"default": None, "options": [None] + list(range(2, 51))},
        "min_impurity_decrease": {"default": 0.0, "options": [i/1000 for i in range(0, 101)]},
        "class_weight": {"default": None, "options": [None, "balanced"]},
        "ccp_alpha": {"default": 0.0, "options": [i/1000 for i in range(0, 101)]},
    },

    "Random Forest": {
        "n_estimators": {"default": 100, "options": list(range(10, 1001, 10))},
        "criterion": {"default": "gini", "options": ["gini", "entropy", "log_loss"]},
        "max_depth": {"default": None, "options": [None] + list(range(1, 51))},
        "min_samples_split": {"default": 2, "options": list(range(2, 21))},  # only ints
        "min_samples_leaf": {"default": 1, "options": list(range(1, 21))},  # only ints
        "min_weight_fraction_leaf": {"default": 0.0, "options": [i/100 for i in range(0, 51)]},
        "max_features": {"default": "sqrt", "options": ["sqrt", "log2", None]},
        "bootstrap": {"default": True, "options": [True, False]},
        "oob_score": {"default": False, "options": [True, False]},
        "n_jobs": {"default": None, "options": [None] + list(range(-1, 33))},
        "random_state": {"default": None, "options": [None] + list(range(0, 101))},
        "verbose": {"default": 0, "options": list(range(0, 4))},
        "warm_start": {"default": False, "options": [True, False]},
        "class_weight": {"default": None, "options": [None, "balanced"]},
        "ccp_alpha": {"default": 0.0, "options": [i/1000 for i in range(0, 101)]},
        "max_samples": {"default": None, "options": [None] + list(range(1, 101))},  # only ints
    },

    "SVM": {
        "C": {"default": 1.0, "options": [i/10 for i in range(1, 101)]},
        "kernel": {"default": "rbf", "options": ["linear", "poly", "rbf", "sigmoid", "precomputed"]},
        "degree": {"default": 3, "options": list(range(0, 11))},
        "gamma": {"default": "scale", "options": ["scale", "auto"]},
        "coef0": {"default": 0.0, "options": [i/10 for i in range(-10, 11)]},
        "shrinking": {"default": True, "options": [True, False]},
        "probability": {"default": False, "options": [True, False]},
        "tol": {"default": 1e-3, "options": [i/10000 for i in range(1, 101)]},
        "cache_size": {"default": 200, "options": list(range(50, 1001, 50))},
        "class_weight": {"default": None, "options": [None, "balanced"]},
        "verbose": {"default": False, "options": [True, False]},
        "max_iter": {"default": -1, "options": [-1] + list(range(1, 10001))},
        "decision_function_shape": {"default": "ovr", "options": ["ovo", "ovr"]},
        "break_ties": {"default": False, "options": [True, False]},
        "random_state": {"default": None, "options": [None] + list(range(0, 101))},
    },

    "XGBoost": {
        "n_estimators": {"default": 100, "options": list(range(10, 1001, 10))},
        "max_depth": {"default": 6, "options": list(range(1, 21))},
        "learning_rate": {"default": 0.3, "options": [i/100 for i in range(1, 101)]},
        "subsample": {"default": 1.0, "options": [i/100 for i in range(1, 101)]},
        "colsample_bytree": {"default": 1.0, "options": [i/100 for i in range(1, 101)]},
        "colsample_bylevel": {"default": 1.0, "options": [i/100 for i in range(1, 101)]},
        "colsample_bynode": {"default": 1.0, "options": [i/100 for i in range(1, 101)]},
        "gamma": {"default": 0, "options": [i/10 for i in range(0, 21)]},
        "reg_alpha": {"default": 0, "options": [i/10 for i in range(0, 21)]},
        "reg_lambda": {"default": 1, "options": [i/10 for i in range(0, 21)]},
        "min_child_weight": {"default": 1, "options": list(range(0, 21))},  # only ints
        "scale_pos_weight": {"default": 1, "options": list(range(0, 21))},  # only ints
        "booster": {"default": "gbtree", "options": ["gbtree", "gblinear", "dart"]},
        "n_jobs": {"default": None, "options": [None] + list(range(-1, 33))},
        "random_state": {"default": 0, "options": list(range(0, 101))},
        "verbosity": {"default": 1, "options": [0, 1, 2, 3]},
    },

    "AdaBoost": {
        "n_estimators": {"default": 50, "options": list(range(10, 501, 10))},
        "learning_rate": {"default": 1.0, "options": [i/10 for i in range(1, 101)]},
        "algorithm": {"default": "SAMME.R", "options": ["SAMME", "SAMME.R"]},
        "random_state": {"default": None, "options": [None] + list(range(0, 101))},
    },

    "Gradient Boost": {
        "loss": {"default": "log_loss", "options": ["log_loss", "exponential"]},
        "learning_rate": {"default": 0.1, "options": [i/100 for i in range(1, 101)]},
        "n_estimators": {"default": 100, "options": list(range(10, 1001, 10))},
        "subsample": {"default": 1.0, "options": [i/100 for i in range(1, 101)]},
        "criterion": {"default": "friedman_mse", "options": ["friedman_mse", "squared_error"]},
        "min_samples_split": {"default": 2, "options": list(range(2, 21))},  # only ints
        "min_samples_leaf": {"default": 1, "options": list(range(1, 21))},  # only ints
        "min_weight_fraction_leaf": {"default": 0.0, "options": [i/100 for i in range(0, 51)]},
        "max_depth": {"default": 3, "options": [None] + list(range(1, 51))},
        "min_impurity_decrease": {"default": 0.0, "options": [i/1000 for i in range(0, 101)]},
        "max_features": {"default": None, "options": ["sqrt", "log2", None]},
        "max_leaf_nodes": {"default": None, "options": [None] + list(range(1, 51))},
        "random_state": {"default": None, "options": [None] + list(range(0, 101))},
        "warm_start": {"default": False, "options": [True, False]},
        "ccp_alpha": {"default": 0.0, "options": [i/1000 for i in range(0, 101)]},
    },

    "Logistic Regression": {
        "penalty": {"default": "l2", "options": ["l1", "l2", "elasticnet", None]},
        "dual": {"default": False, "options": [True, False]},
        "tol": {"default": 1e-4, "options": [i/100000 for i in range(1, 101)]},
        "C": {"default": 1.0, "options": [i/10 for i in range(1, 101)]},
        "fit_intercept": {"default": True, "options": [True, False]},
        "intercept_scaling": {"default": 1.0, "options": [i/10 for i in range(1, 101)]},
        "class_weight": {"default": None, "options": [None, "balanced"]},
        "random_state": {"default": None, "options": [None] + list(range(0, 101))},
        "solver": {"default": "lbfgs", "options": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]},
        "max_iter": {"default": 100, "options": list(range(10, 1001, 10))},
        "multi_class": {"default": "auto", "options": ["auto", "ovr", "multinomial"]},
        "verbose": {"default": 0, "options": list(range(0, 4))},
        "warm_start": {"default": False, "options": [True, False]},
        "l1_ratio": {"default": None, "options": [None] + list(range(0, 11))},  # only ints
    },
}


