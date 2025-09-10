from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import gdown

app = Flask(__name__)
CORS(app)  # 👈 Enables CORS for all routes and all origins

BASE_DIR = "models/drug_recommender"
os.makedirs(BASE_DIR, exist_ok=True)

# Google Drive direct download links
SIMILARITY_URL = "https://drive.google.com/uc?id=1Sy2WerOx9RSaQGzEqh-0dhJwgma_HhbX"
MEDICINE_DICT_URL = "https://drive.google.com/uc?id=1CW8Hn9tLaditNXzu9W-b6C5HEAZvu6Z1"

# Local paths
SIMILARITY_PATH = os.path.join(BASE_DIR, "similarity.pkl")
MEDICINE_DICT_PATH = os.path.join(BASE_DIR, "medicine_dict.pkl")


def download_file_if_missing(url, path):
    """Download file from Google Drive if it does not exist locally"""
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} from Google Drive...")
        gdown.download(url, path, quiet=False)
    else:
        print(f"Found {os.path.basename(path)} locally, skipping download.")


# ✅ Ensure required files exist
download_file_if_missing(SIMILARITY_URL, SIMILARITY_PATH)
download_file_if_missing(MEDICINE_DICT_URL, MEDICINE_DICT_PATH)

# ✅ Load models
medicines_dict = pickle.load(open(MEDICINE_DICT_PATH, "rb"))
medicines = pd.DataFrame(medicines_dict)

similarity = pickle.load(open(SIMILARITY_PATH, "rb"))


def recommend(medicine):
    if medicine not in medicines['Drug_Name'].values:
        return []  # return empty if medicine not found
    
    medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(
        list(enumerate(distances)), 
        reverse=True, 
        key=lambda x: x[1]
    )[1:6]

    recommended_medicines = []
    for i in medicines_list:
        drug = medicines.iloc[i[0]].Drug_Name
        recommended_medicines.append(drug)

    # ✅ Return only unique medicines, preserving order
    return list(dict.fromkeys(recommended_medicines))


@app.route('/recommend', methods=['POST'])
def recommend_api():
    data = request.get_json()

    if not data or "medicine" not in data:
        return jsonify({"error": "Please provide a medicine name"}), 400

    medicine_name = data["medicine"]
    recommendations = recommend(medicine_name)

    return jsonify({
        "medicine": medicine_name,
        "recommendations": recommendations
    })


@app.route('/medicines', methods=['GET'])
def medicines_api():
    return jsonify({
        "medicines": medicines['Drug_Name'].tolist()
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default to 5000 locally
    app.run(host="0.0.0.0", port=port, debug=False)
