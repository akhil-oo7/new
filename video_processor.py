import cv2
import numpy as np
from tqdm import tqdm
import os
import logging
import psutil

class VideoProcessor:
    def __init__(self, frame_interval=30, target_size=(224, 224)):
        """
        Initialize the VideoProcessor with deployment-friendly settings.
        
        Args:
            frame_interval (int): Number of frames to skip between extractions (default: 30)
            target_size (tuple): Target size for frame resizing (height, width) (default: (224, 224))
        """
        self.frame_interval = max(1, frame_interval)  # Ensure at least 1
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
        
        # Verify OpenCV backend (important for Render)
        self._verify_opencv()

    def _verify_opencv(self):
        """Verify OpenCV is installed."""
        try:
            version = cv2.__version__
            self.logger.info(f"OpenCV version: {version}")
        except AttributeError:
            raise ImportError("OpenCV not properly installed")

    def _validate_video(self, video_path):
        """Validate the video file exists and is accessible."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if os.path.getsize(video_path) == 0:
            raise ValueError("Video file is empty")

    def extract_frames(self, video_path, max_frames=None):
        """
        Extract frames from a video file with robust error handling and memory optimization.
        
        Args:
            video_path (str): Path to the video file
            max_frames (int): Maximum number of frames to extract (for memory management)
            
        Returns:
            list: List of extracted frames as numpy arrays
            
        Raises:
            ValueError: If video cannot be processed
        """
        self._validate_video(video_path)
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file - may be corrupt or unsupported format")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError("Video contains no frames")
            
            # Adjust frame interval if video is very short
            effective_interval = self.frame_interval
            if total_frames < 100:
                effective_interval = max(1, self.frame_interval // 3)
            
            with tqdm(total=min(total_frames, max_frames*effective_interval) if max_frames else total_frames, 
                     desc="Extracting frames") as pbar:
                frame_count = 0
                extracted_count = 0
                
                while True:
                    if max_frames and extracted_count >= max_frames:
                        break
                        
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % effective_interval == 0:
                        try:
                            # Check memory usage
                            memory_usage = psutil.virtual_memory().percent
                            if memory_usage > 80:  # Threshold for memory usage
                                self.logger.warning("High memory usage detected. Pausing frame extraction.")
                                break
                            
                            # Convert BGR to RGB and resize
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_resized = cv2.resize(frame_rgb, self.target_size)
                            frames.append(frame_resized)
                            extracted_count += 1
                        except Exception as e:
                            self.logger.warning(f"Error processing frame {frame_count}: {str(e)}")
                    
                    frame_count += 1
                    pbar.update(1)
            
            if not frames:
                raise ValueError("No frames were successfully extracted")
                
            return frames
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {str(e)}")
            raise ValueError(f"Video processing error: {str(e)}")
            
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

    @staticmethod
    def cleanup():
        """Clean up OpenCV resources."""
        cv2.destroyAllWindows()