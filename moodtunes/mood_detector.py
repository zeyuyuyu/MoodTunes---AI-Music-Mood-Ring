import cv2
import numpy as np
from fer import FER
from typing import Dict, Optional

class MoodDetector:
    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.cap = None
        self._last_emotion = None

    def start_webcam(self) -> None:
        """Initialize webcam capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError('Failed to access webcam')

    def stop_webcam(self) -> None:
        """Release webcam resources"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()

    def get_current_mood(self) -> Optional[Dict[str, float]]:
        """
        Capture and analyze current facial emotion
        Returns: Dict with emotion probabilities or None if no face detected
        """
        if not self.cap:
            self.start_webcam()

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Analyze frame for emotions
        emotions = self.detector.detect_emotions(frame)
        
        if not emotions:
            return self._last_emotion

        # Get emotions from strongest detected face
        dominant_face = max(emotions, key=lambda x: x['emotions']['happy'])
        self._last_emotion = dominant_face['emotions']
        
        return self._last_emotion

    def get_dominant_emotion(self) -> Optional[str]:
        """Returns the strongest detected emotion"""
        emotions = self.get_current_mood()
        if not emotions:
            return None
            
        return max(emotions.items(), key=lambda x: x[1])[0]

    def __enter__(self):
        self.start_webcam()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_webcam()
