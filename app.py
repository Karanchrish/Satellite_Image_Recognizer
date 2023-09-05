import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the pre-trained model
model = load_model('C:/Users/karan/Downloads/DL/Karan/model.h5')

# Define class names
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Streamlit app title and description
st.title("Satellite Image Recognizer")
st.write("Upload an image to predict the given Image")

# Function to make predictions
def predict_image(image):
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.reshape(image_array, (1, 255, 255, 3))
    
    predictions = model.predict(image_array)
    class_index = np.argmax(predictions[0])
    predicted_label = class_names[class_index]
    
    return predicted_label

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
    # Make predictions when the user clicks the "Predict" button
    if st.button("Predict"):
        image = load_img(uploaded_image, target_size=(255, 255))
        predicted_label = predict_image(image)
        
        # Display the prediction result
        st.write("Prediction:", predicted_label)

# Optional: Add a sidebar with additional information or options
# st.sidebar.title("Sidebar Title")
# st.sidebar.write("Add content here")
