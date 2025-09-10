from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import gdown

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
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





class MedicineStockPredictor:
    def __init__(self):
        self.medicine_categories = {
            'paracetamol': 'Pain Killer',
            'ibuprofen': 'Pain Killer',
            'amoxicillin': 'antibiotics',
            'metformin': 'Diabetes',
        }

        self.category_parameters = {
            'common': {'lead_time': 3, 'demand_factor': 1.2},
            'Pain Killer': {'lead_time': 4, 'demand_factor': 1.8},
            'antibiotics': {'lead_time': 7, 'demand_factor': 1.3},
            'Diabetes': {'lead_time': 5, 'demand_factor': 1.6},
        }

    def categorize_medicine(self, medicine_name):
        if pd.isna(medicine_name):
            return 'common'
        medicine_lower = str(medicine_name).lower()
        for keyword, category in self.medicine_categories.items():
            if keyword in medicine_lower:
                return category
        return 'common'

    def predict_stock(self, medicine_name, current_stock, price, category=None):
        current_stock = current_stock if not pd.isna(current_stock) else 0
        price = price if not pd.isna(price) else 50
        category = self.categorize_medicine(medicine_name) if not category else category
        params = self.category_parameters.get(category, self.category_parameters['common'])

        base_demand = 20
        price_factor = max(0.5, 1 - (price / 1000))
        daily_demand = base_demand * price_factor * params['demand_factor']

        days_of_stock = current_stock / daily_demand if daily_demand > 0 else 999
        lead_time = params['lead_time']
        lead_time_demand = daily_demand * lead_time
        safety_stock = daily_demand * 1.5
        reorder_point = lead_time_demand + safety_stock
        needs_reorder = current_stock <= reorder_point
        reorder_quantity = max(daily_demand * 30 - current_stock, 0)

        if days_of_stock < 3:
            urgency = "Critical"
        elif days_of_stock < 7:
            urgency = "High"
        elif days_of_stock < 14:
            urgency = "Medium"
        else:
            urgency = "Low"

        return {
            'medicine_name': medicine_name,
            'category': category,
            'current_stock': current_stock,
            'price': price,
            'daily_demand_estimate': round(daily_demand, 2),
            'days_of_stock': round(days_of_stock, 2),
            'reorder_point': round(reorder_point, 2),
            'reorder_quantity': round(reorder_quantity, 2),
            'needs_reorder': needs_reorder,
            'urgency': urgency,
            'lead_time_days': lead_time
        }

    def predict_from_excel(self, file, medicine_col='medicine_name', stock_col='stock', price_col='price', category_col=None):
        df = pd.read_excel(file)
        required_cols = [medicine_col, stock_col, price_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in Excel file.")

        results = []
        for _, row in df.iterrows():
            result = self.predict_stock(
                row[medicine_col],
                row[stock_col],
                row[price_col],
                row[category_col] if category_col and category_col in df.columns else None
            )
            results.append(result)
        return results


# --- Flask API ---
predictor = MedicineStockPredictor()

@app.route('/predict-stock', methods=['POST'])
def predict_stock_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.split('.')[-1] in ALLOWED_EXTENSIONS:
        try:
            results = predictor.predict_from_excel(file)
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400






if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default to 5000 locally
    app.run(host="0.0.0.0", port=port, debug=False)
