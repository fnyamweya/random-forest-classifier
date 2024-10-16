import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(model, feature_names):
    """
    Plot the feature importance of the trained model.
    :param model: Trained RandomForestClassifier model
    :param feature_names: List of feature names
    """
    print("Plotting feature importance...")

    # Get feature importance values from the model
    feature_importance = model.feature_importances_

    # Create a DataFrame for better readability
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })

    # Sort the DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="viridis")
    plt.title('Feature Importance in Random Forest Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show()

    print("Feature importance plot generated successfully.")
