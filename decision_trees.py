import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import tree
import joblib

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    target_column = df.columns[-1]  # Automatically selects the last column as the target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X.columns, target_column

def initialize_model(task_type='classification'):
    if task_type == 'classification':
        return DecisionTreeClassifier(random_state=42)
    else:
        return DecisionTreeRegressor(random_state=42)

def hyperparameter_tuning(clf, X_train, y_train, task_type='classification'):
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # For regression, we use KFold (not StratifiedKFold)
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # For regression, use an appropriate scoring metric like neg_mean_squared_error
    if task_type == 'classification':
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kfold, scoring='accuracy')
    else:
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error')
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(clf_best, X_test, y_test, task_type='classification'):
    if task_type == 'classification':
        y_pred = clf_best.predict(X_test)
        return accuracy_score(y_test, y_pred)
    else:
        y_pred = clf_best.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return np.sqrt(mse)

def visualize_tree(clf_best, feature_names, task_type='classification'):
    plt.figure(figsize=(20, 10))
    
    # Tree Plot Styling
    if task_type == 'classification':
        tree.plot_tree(clf_best, filled=True, feature_names=feature_names, class_names=clf_best.classes_, rounded=True, proportion=False, fontsize=12)
    else:
        tree.plot_tree(clf_best, filled=True, feature_names=feature_names, rounded=True, proportion=False, fontsize=12)

    # Customizing the plot to make it more user-friendly
    plt.title("Decision Tree Visualization", fontsize=16)
    
    plt.show()

def get_custom_tree_labels(clf_best, task_type='classification'):
    # Extract decision tree structure
    tree_structure = clf_best.tree_
    
    # Custom labels for nodes
    custom_labels = {}
    
    # Iterate through all the nodes in the tree to apply custom terminology
    for i in range(tree_structure.node_count):
        # If it's a leaf node (Final Answer)
        if tree_structure.children_left[i] == tree_structure.children_right[i]:
            if task_type == 'classification':
                custom_labels[i] = f"Final Answer: {clf_best.classes_[tree_structure.value[i].argmax()]}"
            else:
                custom_labels[i] = f"Final Answer: {tree_structure.value[i].mean():.2f}"
        # If it's a decision node (Decision Points)
        else:
            if task_type == 'classification':
                custom_labels[i] = f"Decision Point: {tree_structure.feature_names_in_[tree_structure.feature[i]]}?" 
            else:
                custom_labels[i] = f"Decision Point: {tree_structure.feature_names_in_[tree_structure.feature[i]]}?"

    return custom_labels


def visualize_tree_with_custom_labels(clf_best, feature_names, task_type='classification'):
    # Create a figure for the visualization
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot the tree
    if task_type == 'classification':
        tree.plot_tree(clf_best, filled=True, feature_names=feature_names, class_names=clf_best.classes_, rounded=True, fontsize=12, ax=ax)
    else:
        tree.plot_tree(clf_best, filled=True, feature_names=feature_names, rounded=True, fontsize=12, ax=ax)
    
    # Get custom labels
    custom_labels = get_custom_tree_labels(clf_best, task_type)
    
    # Adding the custom labels on top of the plot
    for node, label in custom_labels.items():
        # Adjust xy coordinates if needed to fine-tune the position
        ax.annotate(label, xy=(tree_structure.node_x[node], tree_structure.node_y[node]),
                    xycoords='figure fraction', horizontalalignment='center', fontsize=10, color='black')
    
    # Set title and show the plot
    ax.set_title("Decision Tree: Path to Final Answer", fontsize=16)
    plt.show()

def save_and_load_model(clf_best, model_filename='decision_tree_model.pkl'):
    joblib.dump(clf_best, model_filename)
    clf_loaded = joblib.load(model_filename)
    return clf_loaded

def train_decision_tree(task_type='classification', file_path=None):
    # Load and preprocess data based on the task type
    if not file_path:
        file_path = 'classification_data.csv' if task_type == 'classification' else 'regression_data.csv'
    
    X_train, X_test, y_train, y_test, feature_names, target_column = load_and_preprocess_data(file_path)
    clf = initialize_model(task_type)
    clf_best = hyperparameter_tuning(clf, X_train, y_train, task_type)
    clf_best.fit(X_train, y_train)
    
    performance = evaluate_model(clf_best, X_test, y_test, task_type)
    print(f'Performance: {performance:.2f}')
    
    visualize_tree(clf_best, feature_names, task_type)
    
    clf_loaded = save_and_load_model(clf_best)
    performance_loaded = evaluate_model(clf_loaded, X_test, y_test, task_type)
    print(f'Performance (loaded model): {performance_loaded:.2f}')
    
    return clf_best
