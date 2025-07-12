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
# Define the path to the locally saved model
LOCAL_MODEL_PATH = "./layoutlm-local-model"

try:
    logging.info(f"Loading model from local path: {LOCAL_MODEL_PATH}")
    # Update the pipeline to load from the local directory
    extractor_pipeline = pipeline(
        "document-question-answering",
        model=LOCAL_MODEL_PATH,
        tokenizer=LOCAL_MODEL_PATH  # Explicitly load the tokenizer as well
    )
    logging.info("Model loaded successfully from local path.")
except Exception as e:
    logging.error(f"Failed to load model from {LOCAL_MODEL_PATH}: {e}")
    extractor_pipeline = None


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


@app.route('/extract', methods=['POST'])
def extract_invoice_data():
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
    # Use the PORT environment variable provided by Azure, default to 8000 for local dev
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
