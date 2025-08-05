from flask import Flask, render_template, request
from pyngrok import ngrok
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)
model = load_model("digit_model.h5")
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    digit = None
    confidence = None
    if request.method == "POST":
        file = request.files["image"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], "result.png")
        file.save(path)
        img = preprocess_image(path)
        prediction = model.predict(img)
        digit = int(np.argmax(prediction))
        confidence = round(float(np.max(prediction)) * 100, 2)
    return render_template("index.html", digit=digit, confidence=confidence)

# Start the Flask app and tunnel
if __name__ == "__main__":
    public_url = ngrok.connect(5000)
    print("Your app is live at:", public_url)
    app.run(port=5000)
