from flask import Flask, request, jsonify
from model import load_model
from utils import preprocess_features

app = Flask(__name__)

# Load the trained Random Forest model
model = load_model('random_forest_iris_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the species of Iris based on input features sent via POST request.
    Expected input format:
    {
        "features": [sepal_length, sepal_width, petal_length, petal_width]
    }
    """
    data = request.get_json()
    features = data.get('features', None)

    if features is None or len(features) != 4:
        return jsonify({'error': 'Invalid input. Please provide all 4 features.'}), 400

    # Preprocess features and make prediction
    prediction = model.predict([features])[0]
    confidence = model.predict_proba([features])[0]
    
    species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    predicted_species = species_mapping[prediction]

    return jsonify({
        'species': predicted_species,
        'confidence': dict(zip(species_mapping.values(), confidence))
    })

if __name__ == '__main__':
    app.run(debug=True)
