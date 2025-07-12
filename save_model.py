from transformers import pipeline
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the model name and the directory to save it in
MODEL_NAME = "impira/layoutlm-document-qa"
SAVE_DIRECTORY = "layoutlm-local-model"


def save_model_locally():
    """
    Downloads the specified model from Hugging Face and saves it to a local directory.
    """
    try:
        if os.path.exists(SAVE_DIRECTORY) and os.listdir(SAVE_DIRECTORY):
            logging.warning(f"Directory '{SAVE_DIRECTORY}' already exists and is not empty. Skipping download.")
            logging.info("If you need to re-download the model, please delete the directory first.")
            return

        logging.info(f"Downloading model: {MODEL_NAME}")
        # Create a pipeline instance. This triggers the download.
        extractor_pipeline = pipeline(
            "document-question-answering",
            model=MODEL_NAME
        )

        logging.info(f"Saving model and tokenizer to '{SAVE_DIRECTORY}'...")
        # The pipeline object doesn't have a single save method, so we save its components.
        extractor_pipeline.model.save_pretrained(SAVE_DIRECTORY)
        extractor_pipeline.tokenizer.save_pretrained(SAVE_DIRECTORY)

        logging.info("Model saved successfully!")

    except Exception as e:
        logging.error(f"An error occurred while saving the model: {e}")


if __name__ == '__main__':
    save_model_locally()
