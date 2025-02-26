import io
import os
import numpy as np
import tensorflow as tf
#from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request, jsonify
from PIL import Image

# Load the TensorFlow SavedModel (the directory "model_dir" should contain saved_model.pb and the variables folder)
MODEL_DIR = "./models/model_checkpoints"
model = tf.keras.models.load_model(MODEL_DIR)

app = Flask(__name__)

def prepare_image(image, target_size=(224, 224)):
    # Convert to RGB if needed and resize
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    
    # Convert to array and normalize to [0, 1]
    image = img_to_array(image)  # shape: (224, 224, 3)
    image = image / 255.0         # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


@app.route('/predict', methods=['POST'])
def predict():
    # Ensure a file is provided in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    
    try:
        image = Image.open(file.stream)
    except Exception as e:
        return jsonify({'error': 'Invalid image file'}), 400

    # Preprocess the image for the model
    processed_image = prepare_image(image)

    # Run prediction
    preds = model.predict(processed_image)
    predicted_class = int(np.argmax(preds, axis=1)[0])
    
    # Return the prediction result as JSON
    return jsonify({'predicted_class': predicted_class, 'probabilities': preds.tolist()})

if __name__ == '__main__':
    # Cloud Run expects the container to listen on port 8080
    port = int(os.environ.get('PORT', 8080))

    app.run(host='0.0.0.0', port=port)
