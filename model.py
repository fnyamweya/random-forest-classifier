import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Set up logging
logging.basicConfig(filename='model_performance.log', level=logging.INFO)

def perform_grid_search(X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.
    :param X_train: Features for training
    :param y_train: Labels for training
    :return: Best trained model
    """
    print("Initializing Random Forest model...")
    rf = RandomForestClassifier(random_state=42)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    print("Starting Grid Search for hyperparameter tuning...")
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print(f"Grid Search completed. Best Parameters: {grid_search.best_params_}")
    logging.info(f"Grid Search Best Parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using a confusion matrix and classification report.
    :param model: Trained RandomForestClassifier
    :param X_test: Test features
    :param y_test: Test labels
    """
    print("Starting model evaluation...")
    y_pred = model.predict(X_test)
    
    # Print the classification report
    print("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # Print the confusion matrix
    print("Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    logging.info(f"Classification Report: {report}")
    logging.info(f"Confusion Matrix: {conf_matrix}")
    
    print("Model evaluation completed successfully.")

def cross_validate_model(model, X_train, y_train):
    """
    Perform cross-validation and print average accuracy.
    :param model: RandomForestClassifier model
    :param X_train: Training features
    :param y_train: Training labels
    """
    print("Starting cross-validation...")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    avg_score = scores.mean() * 100
    print(f"Cross-Validation Accuracy: {avg_score:.2f}%")
    logging.info(f"Cross-Validation Accuracy: {avg_score:.2f}%")
    print("Cross-validation completed.")

def save_model(model, file_path='random_forest_iris_model.pkl'):
    """
    Save the trained Random Forest model to a file.
    :param model: Trained RandomForestClassifier model
    :param file_path: Path to save the model
    """
    joblib.dump(model, file_path)
    print(f"Model saved successfully as {file_path}")
    logging.info(f"Model saved as {file_path}")

def load_model(file_path='random_forest_iris_model.pkl'):
    """
    Load a trained Random Forest model from a file.
    :param file_path: Path to the model file
    :return: Loaded RandomForestClassifier model
    """
    model = joblib.load(file_path)
    print(f"Model loaded successfully from {file_path}")
    return model
