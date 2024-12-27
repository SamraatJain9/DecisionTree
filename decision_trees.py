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
    plt.figure(figsize=(20,10))
    if task_type == 'classification':
        tree.plot_tree(clf_best, filled=True, feature_names=feature_names, class_names=clf_best.classes_, rounded=True)
    else:
        tree.plot_tree(clf_best, filled=True, feature_names=feature_names, rounded=True)
    plt.show()

def save_and_load_model(clf_best, model_filename='decision_tree_model.pkl'):
    joblib.dump(clf_best, model_filename)
    clf_loaded = joblib.load(model_filename)
    return clf_loaded

def main():
    file_path = 'regression_data.csv'  # Replace with your dataset path
    task_type = 'regression'  # Change to 'classification' for classification tasks

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

if __name__ == '__main__':
    main()
