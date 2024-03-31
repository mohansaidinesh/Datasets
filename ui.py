import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from functools import partial
import pandas as pd

# Specify the encoding explicitly when reading the CSV file
df = pd.read_csv("Plants_Desc.csv", encoding='latin1')

# Define a function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch dimension
    return img_array

# Define a function to make predictions on an image
def predict_image(image_path, model):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    # Make predictions
    predictions = model.predict(img_array)
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)
    return predicted_class, predictions[0]

# Define the ResidualUnit class
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="same", kernel_initializer="he_normal",
                        use_bias=False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

# Register the custom layer ResidualUnit
custom_objects = {'ResidualUnit': ResidualUnit}

# Load the saved model with custom objects scope
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model('model.h5')

# Define class indices
class_indices = {
    'Aloevera': 0,
    'Amla': 1,
    'Bamboo': 2,
    'Beans': 3,
    'Betel': 4,
    'Chilly': 5,
    'Coffee': 6,
    'Coriender': 7,
    'Drumstick': 8,
    'Ganigale': 9,
    'Ginger': 10,
    'Guava': 11,
    'Henna': 12,
    'Hibiscus': 13,
    'Jasmine': 14,
    'Lemon': 15,
    'Mango': 16,
    'Marigold': 17,
    'Mint': 18,
    'Neem': 19,
    'Onion': 20,
    'Palak': 21,
    'Papaya': 22,
    'Parijatha': 23,
    'Pea': 24,
    'Pomoegranate': 25,
    'Pumpkin': 26,
    'Raddish': 27,
    'Rose': 28,
    'Sampige': 29,
    'Sapota': 30,
    'Seethapala': 31,
    'Spinach1': 32,
    'Tamarind': 33,
    'Tomato': 34,
    'Tulsi': 35,
    'Turmeric': 36,
    'ashoka' : 37,
    'camphor' : 38
}

# Streamlit UI
st.title('Medicinal Leaf Prediction')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the selected image
    st.image(uploaded_file, caption='Uploaded Image.', width=200)
    # Make predictions on the image
    predicted_class_index, probabilities = predict_image(uploaded_file, model)
    # Map the predicted class index to class label
    predicted_class_label = [key for key, value in class_indices.items() if value == predicted_class_index][0]
    # Display the predicted class label and probabilities
    st.write("Plant is :", predicted_class_label)
    st.divider()
    st.markdown('**Description**')
    res = df[df['Type'].str.contains(predicted_class_label, case=False,na=False)]
    for index,row in res.iterrows():
        st.write(row['Description'])
    st.divider()
    st.markdown('**Uses**')
    ans=df[df['Type'].str.contains(predicted_class_label, case=False,na=False)]
    for index,row in res.iterrows():
        st.write(row['Uses'])
