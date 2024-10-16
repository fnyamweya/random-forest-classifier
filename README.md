# Random Forest Classifier

This project implements a sophisticated **Random Forest Classifier** using the **Iris dataset**. The classifier includes:

- Hyperparameter tuning with **GridSearchCV**.
- Feature scaling and preprocessing.
- Cross-validation to ensure robust model performance.
- Detailed evaluation metrics (confusion matrix, precision, recall, F1-score).
- Feature importance visualization.

## Features

- **Dataset**: Uses the Iris dataset. If not available, it generates it automatically.
- **Grid Search**: Automatically tunes hyperparameters for optimal model performance.
- **Visualization**: Plots feature importance for better model interpretability.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:fnyamweya/random-forest-classifier.git
cd andom-forest-classifier
```

### 2. Create and Activate a Virtual Environment

Create a virtual environment to isolate project dependencies:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment (Linux/macOS)
source venv/bin/activate

# Activate the virtual environment (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

Once the virtual environment is activated, install the dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run the Project

Run the main script:

```bash
python main.py
```

### 5. Deactivating the Virtual Environment

When you're done, deactivate the virtual environment with:

```bash
deactivate
```

#### Requirements

The project requires the following Python packages, which can be installed via `requirements.txt`:

- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

### 6. Updating Dependencies

To add new dependencies, install them using pip and update requirements.txt:

```bash
pip freeze > requirements.txt
```
