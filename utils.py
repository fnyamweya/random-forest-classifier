import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load dataset from the file path or generate it if the file doesn't exist or is empty.
    If the file is missing or empty, the Iris dataset is generated and saved to the file path.
    :param file_path: string, path to the CSV file.
    :return: pandas DataFrame
    """
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        print(f"{file_path} not found or is empty. Generating Iris dataset...")
        iris = load_iris()
        data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        data['species'] = iris.target
        # Save to CSV for future use
        data.to_csv(file_path, index=False)
        print(f"Iris dataset saved as {file_path}")
    else:
        print(f"Loading dataset from {file_path}")
        data = pd.read_csv(file_path)
    
    print(f"Dataset loaded successfully from {file_path}")
    return data

def preprocess_data(df, target_column='species'):
    """
    Preprocess the dataset: Scale features and separate into features (X) and target (y).
    :param df: DataFrame, input dataset.
    :param target_column: string, the column name of the target variable (default is 'species').
    :return: tuple of scaled features (X) and labels (y).
    """
    print("Starting data preprocessing...")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Data preprocessing completed successfully.")
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    :param X: Features (scaled).
    :param y: Target labels.
    :param test_size: float, proportion of data to use for testing (default 0.2).
    :param random_state: int, seed for random splitting (default 42).
    :return: tuple of training and testing sets (X_train, X_test, y_train, y_test).
    """
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("Data split completed.")
    return X_train, X_test, y_train, y_test
