import os
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from transformers import pipeline
import torch
from dotenv import load_dotenv

# Disable GPU if necessary (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Uncomment to force CPU-only mode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ===== Configuration =====
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'dev-fallback-key'),
    UPLOAD_FOLDER=os.getenv('UPLOAD_FOLDER', 'uploads'),
    MAX_CONTENT_LENGTH=int(os.getenv('MAX_CONTENT_MB', 100)) * 1024 * 1024,
    ALLOWED_EXTENSIONS={'mp4', 'avi', 'mov'},
    MODEL_PATH=os.getenv('MODEL_PATH', 'models/best_model')
)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_PATH'], exist_ok=True)

# Preload the model at startup
model = pipeline("image-classification", model=app.config['MODEL_PATH'])  # Adjust pipeline and model path as needed

# ===== Helper Functions =====
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_frames(video):
    # Dummy frame extraction logic for demonstration
    return [torch.zeros((3, 224, 224))]  # Replace with actual frame extraction logic

# ===== Routes =====
@app.route('/')
def home():
    return "App is running!"

@app.route('/analyze', methods=['POST'])
def analyze_video():
    logger.info("Received video for analysis")
    # Validate request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not video or not allowed_file(video.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Secure file handling
        filename = secure_filename(video.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(filepath)

        # Example: Extract frames and run inference
        frames = extract_frames(filepath)  # Replace with your frame extraction logic
        results = [model(frame) for frame in frames]  # Run inference on each frame

        return jsonify(results)

    except Exception as e:
        app.logger.error(f"Analysis failed: {str(e)}")
        return jsonify({'error': 'Video processing failed'}), 500

    finally:
        # Cleanup
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

# ===== Error Handlers =====
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({'error': 'File too large'}), 413

# ===== Health Check =====
@app.route('/health')
def health():
    return jsonify({
        'status': 'ready',
        'model_loaded': os.path.exists(
            os.path.join(app.config['MODEL_PATH'], 'model.safetensors')
        )
    })

# ===== Main =====
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 for local testing
    app.run(host='0.0.0.0', port=port)