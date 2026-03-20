from flask import Flask, request, jsonify, render_template, redirect, session
import os
from model_service import predict_stroke
from database import save_result
from firebase_auth import upload_to_firebase
from gradcam import generate_gradcam

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login_page():
    return render_template("login.html")


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/history")
def history():
    return render_template("history.html")


@app.route("/report")
def report():
    return render_template("report.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "mri" not in request.files or "ct" not in request.files:
        return jsonify({"error": "Upload both MRI and CT images"})

    mri_file = request.files["mri"]
    ct_file = request.files["ct"]

    mri_path = os.path.join(UPLOAD_FOLDER, "mri_" + mri_file.filename)
    ct_path = os.path.join(UPLOAD_FOLDER, "ct_" + ct_file.filename)

    mri_file.save(mri_path)
    ct_file.save(ct_path)

    # AI prediction
    prediction, confidence = predict_stroke(mri_path, ct_path)

    # GradCAM (use MRI for visualization)
    heatmap_path = generate_gradcam(mri_path)

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "heatmap": heatmap_path
    })