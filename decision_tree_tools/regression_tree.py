import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import tree
import joblib
import json

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    target_column = df.columns[-1]  # Automatically selects the last column as the target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X.columns, target_column


def initialize_model():
    """Initialize the DecisionTreeRegressor model."""
    return DecisionTreeRegressor(random_state=42)


def hyperparameter_tuning(regressor, X_train, y_train):
    """Perform hyperparameter tuning to find the best regressor."""
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def evaluate_model(regressor_best, X_test, y_test):
    """Evaluate the performance of the model."""
    y_pred = regressor_best.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return np.sqrt(mse)  # Return RMSE (root mean squared error)


def visualize_tree(regressor_best, feature_names):
    """Visualize the regression tree."""
    plt.figure(figsize=(20, 10))
    tree.plot_tree(regressor_best, filled=True, feature_names=feature_names, rounded=True, fontsize=12)
    plt.title("Decision Tree Regression Visualization", fontsize=16)
    plt.show()


def get_textual_tree_representation_with_lines(regressor_best, feature_names):
    """Generate a textual (ASCII-style) representation of the regression tree with lines."""
    tree_structure = regressor_best.tree_
    feature_names = feature_names.tolist()  # Convert feature names to a list

    def recurse_tree(node, indent=""):
        if tree_structure.children_left[node] == tree_structure.children_right[node]:
            # Leaf node
            value = tree_structure.value[node]
            return f"{indent}└── Leaf: {value.mean():.2f}"  # Display the mean for regression
        else:
            # Decision node
            feature = feature_names[tree_structure.feature[node]]
            threshold = tree_structure.threshold[node]
            left = recurse_tree(tree_structure.children_left[node], indent + "│   ")
            right = recurse_tree(tree_structure.children_right[node], indent + "    ")
            return (f"{indent}├── Decision: {feature} <= {threshold:.2f}?\n"
                    f"{left}\n"
                    f"{right}")
    
    return recurse_tree(0)


def save_tree_as_json(regressor_best, feature_names, filename="regression_tree.json"):
    """Save the regression tree structure as a JSON file."""
    tree_structure = regressor_best.tree_
    tree_dict = {}

    def recurse_tree_to_json(node, parent_dict):
        if tree_structure.children_left[node] == tree_structure.children_right[node]:
            # Leaf node
            parent_dict["leaf"] = tree_structure.value[node].mean()  # Mean value for regression
        else:
            # Decision node
            feature = feature_names[tree_structure.feature[node]]
            threshold = tree_structure.threshold[node]
            parent_dict["question"] = f"{feature} <= {threshold:.2f}"
            parent_dict["yes"] = {}
            parent_dict["no"] = {}
            recurse_tree_to_json(tree_structure.children_left[node], parent_dict["yes"])
            recurse_tree_to_json(tree_structure.children_right[node], parent_dict["no"])

    recurse_tree_to_json(0, tree_dict)

    with open(filename, 'w') as json_file:
        json.dump(tree_dict, json_file, indent=4)
    print(f"Tree saved as {filename}")


def save_and_load_model(regressor_best, model_filename='regression_tree_model.pkl'):
    """Save and load the trained model."""
    joblib.dump(regressor_best, model_filename)
    regressor_loaded = joblib.load(model_filename)
    return regressor_loaded


def train_regression_tree(file_path=None):
    """Main function to train the regression tree."""
    if not file_path:
        file_path = 'regression_data.csv'  # Default file path for regression data
    
    X_train, X_test, y_train, y_test, feature_names, target_column = load_and_preprocess_data(file_path)
    
    regressor = initialize_model()
    regressor_best = hyperparameter_tuning(regressor, X_train, y_train)
    
    regressor_best.fit(X_train, y_train)

    performance = evaluate_model(regressor_best, X_test, y_test)
    print(f"Model performance (RMSE): {performance:.2f}")
    
    visualize_tree(regressor_best, feature_names)
    
    # Generate and display textual representation
    tree_representation = get_textual_tree_representation_with_lines(regressor_best, feature_names)
    print("\nTextual Representation of the Tree:")
    print(tree_representation)

    # Save the tree as JSON
    save_tree_as_json(regressor_best, feature_names)
    
    regressor_loaded = save_and_load_model(regressor_best)
    performance_loaded = evaluate_model(regressor_loaded, X_test, y_test)
    print(f"Performance (loaded model): {performance_loaded:.2f}")
    
    return regressor_best


if __name__ == '__main__':
    train_regression_tree('regression_data.csv')
