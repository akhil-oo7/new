import os
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from video_processor import VideoProcessor
from content_moderator import ContentModerator

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

# ===== Initialize Components =====
video_processor = VideoProcessor()
content_moderator = ContentModerator(train_mode=False)

# ===== Helper Functions =====
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ===== Routes =====
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    logger.info("Received video for analysis")
    # Validate request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Secure file handling
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process video
        frames = video_processor.extract_frames(filepath)
        results = content_moderator.analyze_frames(frames)
        
        # Generate response
        unsafe_frames = [r for r in results if r['flagged']]
        response = {
            'status': 'UNSAFE' if unsafe_frames else 'SAFE',
            'total_frames': len(results),
            'unsafe_frames': len(unsafe_frames),
            'unsafe_percentage': (len(unsafe_frames) / len(results)) * 100 if results else 0,
            'confidence': max(r['confidence'] for r in results) if results else 1.0,
            'details': [
                {
                    'frame': idx,
                    'reason': r['reason'],
                    'confidence': r['confidence']
                } 
                for idx, r in enumerate(results) if r['flagged']
            ]
        }
        
        return jsonify(response)

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