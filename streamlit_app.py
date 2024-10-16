import streamlit as st
from model import load_model

# Load the trained Random Forest model
model = load_model('random_forest_iris_model.pkl')

def main():
    st.title('Iris Flower Species Prediction')

    # User inputs for the 4 Iris flower features
    sepal_length = st.slider('Sepal Length', 4.0, 8.0, 5.0)
    sepal_width = st.slider('Sepal Width', 2.0, 4.5, 3.0)
    petal_length = st.slider('Petal Length', 1.0, 7.0, 1.4)
    petal_width = st.slider('Petal Width', 0.1, 2.5, 0.2)

    if st.button('Predict'):
        # Use model to predict species based on user inputs
        features = [sepal_length, sepal_width, petal_length, petal_width]
        prediction = model.predict([features])[0]
        confidence = model.predict_proba([features])[0]

        species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        predicted_species = species_mapping[prediction]

        st.write(f"Predicted Species: **{predicted_species}**")
        st.write("Confidence Scores:")
        st.write(dict(zip(species_mapping.values(), confidence)))

if __name__ == '__main__':
    main()
