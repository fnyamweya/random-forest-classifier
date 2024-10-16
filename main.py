from utils import load_data, preprocess_data, split_data
from model import perform_grid_search, evaluate_model, cross_validate_model, save_model, load_model
from plots import plot_feature_importance

def main():
    print("Starting the Random Forest Classifier process...")

    # Load the dataset (will generate the iris dataset if not found)
    data = load_data('data/iris.csv')

    # Preprocess the dataset: Split into features (X) and target (y)
    X, y = preprocess_data(data, 'species')

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Data split into training and testing sets.")

    # Perform grid search for hyperparameter tuning
    best_model = perform_grid_search(X_train, y_train)

    # Cross-validate the model
    cross_validate_model(best_model, X_train, y_train)

    # Evaluate the model on test data
    evaluate_model(best_model, X_test, y_test)

    # Save the trained model for later use
    save_model(best_model, 'random_forest_iris_model.pkl')

    # Plot feature importance
    plot_feature_importance(best_model, feature_names=data.columns[:-1])

    print("Random Forest Classifier process completed.")

if __name__ == '__main__':
    main()
