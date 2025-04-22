from transformers import pipeline, AutoModelForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
from flask import Flask, request, jsonify  # Import Flask for web service
import logging  # Add logging for debugging and monitoring
from flask_cors import CORS  # Add CORS for cross-origin requests

class VideoFrameDataset(Dataset):
    def __init__(self, frames, labels, feature_extractor):
        self.frames = frames
        self.labels = labels
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame)
        
        # Preprocess the image
        inputs = self.feature_extractor(image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ContentModerator:
    def __init__(self, model_name="microsoft/resnet-50", train_mode=False):
        """
        Initialize the ContentModerator with a pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            train_mode (bool): Whether to initialize in training mode
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))  # Default to 0.7
        
        # Always use feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        model_path = os.getenv("MODEL_PATH", os.path.join("models", "best_model"))  # Use env variable for model path
        if train_mode:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=2,  # Binary classification: violent vs non-violent
                ignore_mismatched_sizes=True
            ).to(self.device)
        else:
            # Load our trained model
            if os.path.exists(model_path):
                logging.info("Loading trained model...")
                self.model = AutoModelForImageClassification.from_pretrained(
                    model_path,
                    num_labels=2
                ).to(self.device)
                self.model.eval()  # Set to evaluation mode
            else:
                logging.error("Trained model not found. Please train the model first.")
                raise FileNotFoundError("Trained model not found. Please train the model first.")
    
    def analyze_frames(self, frames):
        """
        Analyze frames for inappropriate content.
        
        Args:
            frames (list): List of video frames as numpy arrays
            
        Returns:
            list: List of analysis results for each frame
        """
        if not frames:
            logging.warning("No frames provided for analysis.")
            return []
        
        results = []
        
        # Convert frames to dataset
        dataset = VideoFrameDataset(frames, [0] * len(frames), self.feature_extractor)
        dataloader = DataLoader(dataset, batch_size=32)
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch['pixel_values'].to(self.device)
                outputs = self.model(pixel_values)
                predictions = torch.softmax(outputs.logits, dim=1)
                
                for pred in predictions:
                    # Get probability of violence (class 1)
                    violence_prob = pred[1].item()
                    # Use configurable threshold
                    flagged = violence_prob > self.confidence_threshold
                    
                    results.append({
                        'flagged': flagged,
                        'reason': "Detected violence" if flagged else "No inappropriate content detected",
                        'confidence': violence_prob if flagged else 1 - violence_prob
                    })
        
        return results

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app
moderator = ContentModerator()

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        frames = data.get('frames', [])
        if not frames:
            return jsonify({"error": "No frames provided"}), 400
        
        results = moderator.analyze_frames(frames)
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return jsonify({"error": "Internal server error"}), 500

def validate_env_variables():
    """
    Validate critical environment variables.
    """
    required_vars = ["MODEL_PATH", "PORT"]
    for var in required_vars:
        if not os.getenv(var):
            logging.error(f"Environment variable {var} is not set.")
            raise EnvironmentError(f"Environment variable {var} is required but not set.")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    try:
        validate_env_variables()
        port = int(os.getenv("PORT", 5000))
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logging.critical(f"Application failed to start: {e}")
        exit(1)