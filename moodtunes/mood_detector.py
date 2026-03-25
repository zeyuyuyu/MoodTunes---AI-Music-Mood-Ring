import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MoodDetector:
    def __init__(self, model_path=None):
        if model_path:
            self.model = RandomForestClassifier.load(model_path)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def recommend_music(self, audio_features):
        mood = self.predict(audio_features)[0]
        if mood == 'happy':
            return ['Happy Song 1', 'Happy Song 2', 'Happy Song 3']
        elif mood == 'sad':
            return ['Sad Song 1', 'Sad Song 2', 'Sad Song 3']
        elif mood == 'angry':
            return ['Angry Song 1', 'Angry Song 2', 'Angry Song 3']
        elif mood == 'calm':
            return ['Calm Song 1', 'Calm Song 2', 'Calm Song 3']
        else:
            return []