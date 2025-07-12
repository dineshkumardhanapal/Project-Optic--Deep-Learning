import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from pdf2image import convert_from_path
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask App and CORS
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Machine Learning Model Loading ---
# FIX: Construct an absolute path to the model directory to avoid ambiguity.
# This resolves the "Repo id must use alphanumeric chars" error.
base_dir = os.path.abspath(os.path.dirname(__file__))
LOCAL_MODEL_PATH = os.path.join(base_dir, "layoutlm-local-model")

extractor_pipeline = None
try:
    # Check if the model directory exists before trying to load
    if os.path.exists(LOCAL_MODEL_PATH):
        logging.info(f"Loading model from local path: {LOCAL_MODEL_PATH}")
        extractor_pipeline = pipeline(
            "document-question-answering",
            model=LOCAL_MODEL_PATH,
            tokenizer=LOCAL_MODEL_PATH
        )
        logging.info("Model loaded successfully from local path.")
    else:
        logging.error(f"Model directory not found at: {LOCAL_MODEL_PATH}")

except Exception as e:
    logging.error(f"A critical error occurred while loading the model: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_details_from_image(image):
    """
    Uses the loaded pipeline to ask questions and get answers with their coordinates.
    """
    if not extractor_pipeline or image is None:
        return {"Error": "Model or image not available."}

    questions = {
        "Invoice Number": "What is the invoice number?",
        "Invoice Date": "What is the invoice date?",
        "Vendor Name": "Who is the vendor?",
        "Bill To": "Who is the bill to recipient?",
        "Ship To": "What is the shipping address?",
        "Subtotal": "What is the subtotal?",
        "Discount": "What is the discount amount?",
        "Shipping Cost": "What is the shipping cost or shipping fee?",
        "Total": "What is the total amount?",
        "Balance Due": "What is the balance due?",
        "Order ID": "What is the Order ID?",
        "Ship Mode": "What is the Ship Mode?",
    }
    
    extracted_data = {}
    for key, question in questions.items():
        try:
            result = extractor_pipeline(image=image, question=question)
            if result:
                answer_data = result[0]
                extracted_data[key] = {
                    "answer": answer_data.get('answer', "Not found"),
                    "box": answer_data.get('box', [0, 0, 0, 0])
                }
            else:
                extracted_data[key] = {"answer": "Not found", "box": [0, 0, 0, 0]}
        except Exception as e:
            logging.error(f"Error extracting '{key}': {e}")
            extracted_data[key] = {"answer": "Extraction Error", "box": [0, 0, 0, 0]}
            
    return extracted_data

# ADDED: A health check endpoint to easily verify if the server is running.
@app.route('/', methods=['GET'])
def health_check():
    model_status = "loaded" if extractor_pipeline is not None else "not loaded"
    return jsonify({"status": "ok", "model_status": model_status})


@app.route('/extract', methods=['POST'])
def extract_invoice_data():
    # ADDED: Check if the model is loaded before processing.
    if extractor_pipeline is None:
        return jsonify({"error": "Model is not loaded, the service is unavailable."}), 503

    if 'invoices' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('invoices')
    
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected files"}), 400

    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logging.info(f"Processing file: {filename}")
            
            try:
                images = convert_from_path(filepath, first_page=1, last_page=1)
                if images:
                    invoice_image = images[0]
                    data = extract_details_from_image(invoice_image)
                    results.append({"filename": filename, "data": data})
                else:
                    results.append({"filename": filename, "data": {"Error": "Could not convert PDF."}})
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                results.append({"filename": filename, "data": {"Error": str(e)}})
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
