from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained Keras model
model_path = 'model/cnn_model.h5'
model = load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', error="No file part")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('result.html', error="No selected file")
    
    if file:
        # Save the uploaded file to the uploads directory
        upload_path = os.path.join('static/uploads', file.filename)
        file.save(upload_path)

        # Load and preprocess the image
        test_image = image.load_img(upload_path, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0         # ✅ Normalize
        test_image = np.expand_dims(test_image, axis=0)

        # Predict
        result = model.predict(test_image)

        # Check threshold
        prediction = 'dog' if result[0][0] >= 0.5 else 'cat'


        # Pass prediction and image path to template
        return render_template(
            'result.html',
            prediction=prediction,
            img_url=upload_path
        )


if __name__ == '__main__':
    app.run(debug=True)
