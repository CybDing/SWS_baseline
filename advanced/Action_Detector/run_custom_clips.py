import torch
import numpy as np
import cv2
import os
import glob
from LSTM import newLSTM
from dataset import CNNFeatureExtractor
from PIL import Image

class ActionPredictor:
    def __init__(self, model_path='best_action_classifier.pth'):
        self.device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.sequence_length = 8
        self.action_names = {0: 'scratching', 1: 'eating'}
        
        # Load model
        self.model = newLSTM(576, 512, 2).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Feature extractor
        self.feature_extractor = CNNFeatureExtractor('mobilenet_v3_small', self.device, True)
        print(f"Model loaded! Acc: {checkpoint['val_acc']:.1f}%")
    
    def predict(self, frame_paths):
        """Predict action from frame paths"""
        if len(frame_paths) > self.sequence_length:
            indices = np.linspace(0, len(frame_paths)-1, self.sequence_length, dtype=int)
            frame_paths = [frame_paths[i] for i in indices]
        
        features = [self.feature_extractor.extract_features_from_frame_path(p) for p in frame_paths]
        sequence = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(sequence)
            probs = torch.softmax(outputs, dim=1)[0]
            pred = torch.argmax(outputs).item()
        
        return {
            'action': self.action_names[pred],
            'confidence': probs[pred].item() * 100,
            'all_probs': {self.action_names[i]: probs[i].item() * 100 for i in range(2)}
        }
    
    def predict_video(self, video_path):
        """Predict from video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            if count % 5 == 0:  # Every 5th frame
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
        cap.release()
        
        os.makedirs('temp', exist_ok=True)
        paths = []
        for i, frame in enumerate(frames):
            path = f'temp/frame_{i}.jpg'
            Image.fromarray(frame).save(path)
            paths.append(path)
        
        result = self.predict(paths)
        
        # Cleanup
        for p in paths: os.remove(p)
        os.rmdir('temp')
        return result


if __name__ == "__main__":
    predictor = ActionPredictor()
    
    clip_path = input("Enter clip path (or press Enter for example): ").strip()
    if not clip_path:
        clip_path = "../cat_video_frames/testing/clip_33"
    
    if os.path.exists(clip_path):
        frames = sorted(glob.glob(os.path.join(clip_path, "*.jpg")))
        if frames:
            result = predictor.predict(frames)
            print(f"\n🎯 Result: {result['action']} ({result['confidence']:.1f}%)")
            for action, prob in result['all_probs'].items():
                print(f"   {action}: {prob:.1f}%")
        else:
            print("No frames found!")
    else:
        print("Path not found!")