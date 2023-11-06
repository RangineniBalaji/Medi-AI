from flask import Flask, render_template, request, jsonify
from PIL import Image

from keras.utils import load_img,img_to_array
from keras.models import load_model
from io import BytesIO
import numpy as np

import json
import requests

bt_model = load_model('braintumor.h5')
pneumonia_model = load_model('pneumonia_detection_model.h5')

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/brain_tumor_detection')
def brain_tumor_detection():
    # Add your code to handle brain tumor detection here
    return render_template('brain_tumor.html')
@app.route('/pneumonia_detection')
def pneumonia_detection():
    # Add your code to handle pneumonia detection here
    return render_template("pneumonia.html")

@app.route('/medical_chatbot')
def medical_chatbot():
    return  render_template("medical_chatbot.html")

@app.route('/medical_reminder')
def medical_reminder():
    # Add your code to handle the medical reminder here
    return render_template("medical_reminder.html")

def generate_response(text):
    headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZTdmYjAxNDctNWQ2OC00MTVlLTg1OTMtNzZhYzU4ZGQxYjJlIiwidHlwZSI6ImFwaV90b2tlbiJ9.w4BYXRL_aa_PlhAUExp7hmdwsNL23Sj3E2xVWC1oxl0"}
    url = "https://api.edenai.run/v2/text/generation"
    payload = {
        "providers": "openai,cohere",
        "text": text,
        "temperature": 0.2,
        "max_tokens": 1000
    }
            
    response = requests.post(url, json=payload, headers=headers)
            
    result = json.loads(response.text)
            
    if 'openai' in result and 'generated_text' in result['openai']:
        generated_text = result['openai']['generated_text']
        return generated_text
    else:
        return "Generated text not found in the response."

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.get_json()['message']
    response = generate_response(user_message)
    return jsonify({'response': response})

@app.route('/pneumonia.html', methods=['post'])
def pneumoniaDetection():
    img_size = (224, 224)
    img_file = request.files['image']
    img = load_img(BytesIO(img_file.read()), target_size=img_size)
    img_arr = img_to_array(img) / 255.
    img_arr = np.expand_dims(img_arr, axis=0)
    pred = pneumonia_model.predict(img_arr)
    pred_class = 'Pneumonia' if pred > 0.5 else 'Normal'
    if pred_class == "Pneumonia":
        inf = "You are diagnosed with Pneumonia!"
    else:
        inf = "Your condition is Normal!"
    return render_template('pneumonia.html', data=inf)

@app.route('/brain_tumor.html', methods=['POST'])
def brain_tumor():
    img_size = (224, 224)
    img_file = request.files['image']
    img = load_img(BytesIO(img_file.read()), target_size=img_size)
    img_arr = img_to_array(img) / 255.
    img_arr = np.expand_dims(img_arr, axis=0)
    pred = bt_model.predict(img_arr)
    pred_class = 'bt' if pred > 0.5 else 'Normal'
    pred_c=''
    if pred < 0.5:
        pred_c = 'You are diagnosed with brain tumor'
    else:
        pred_c = 'your condition is normal'
    return render_template("brain_tumor.html", data=pred_c)

if __name__ == '__main__':
    app.run(debug=True)
