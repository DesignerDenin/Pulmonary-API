from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict import predict
import os

app = Flask(__name__)
CORS(app)
app.config['uploads'] = "uploads"

@app.route("/predict", methods=['POST'])
def generate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    folder = 'uploads'

    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, file.filename)
    file.save(file_path)
    file_url = f"{request.host_url}{file_path}"
    result = predict(file_path)

    return jsonify({'result': result, 'url': file_url})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['uploads'], filename)

@app.route("/", methods=['GET'])
def default():
    return "<h1> Welcome <h1>"

if __name__ == "__main__":
    app.run('0.0.0.0', port=5000)