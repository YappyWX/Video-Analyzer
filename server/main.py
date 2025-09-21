from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

from routes.routes import summarize_routes

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__, static_url_path="/uploads", static_folder="uploads")

app = Flask(__name__)
app.register_blueprint(summarize_routes)
CORS(app, origins='*')

@app.route("/images")
def list_images():
    files = os.listdir(UPLOAD_FOLDER)
    image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
    images = [
        f"http://localhost:8080/uploads/{f}"
        for f in files
        if f.lower().endswith(image_extensions)
    ]
    return jsonify(images)

@app.route("/uploads/<path:filename>")
def serve_image(filename):
    full_path = os.path.join(UPLOAD_FOLDER, filename)
    print("Trying to serve:", full_path)
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": 'No file found'}), 400

    file = request.files["file"]

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    return jsonify({"filename": filename})

if __name__ == "__main__":
    app.run(debug=True, port=8080)