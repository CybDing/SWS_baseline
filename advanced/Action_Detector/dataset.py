import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import numpy as np
import os
from torchvision import transforms, models
import glob
from sklearn.preprocessing import StandardScaler
import pickle
from PIL import Image


class CNNFeatureExtractor:
    def __init__(self, model_name='mobilenet_v3_small', device='cpu', use_imagenet_norm=False):
        self.device = device
        self.model_name = model_name
        self.use_imagenet_norm = use_imagenet_norm
        
        if model_name == 'mobilenet_v3_small':
            self.model = models.mobilenet_v3_small(pretrained=True)
            self.feature_dim = 576  # MobileNetV3-Small feature dimension
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=True)
            self.feature_dim = 1280  # MobileNetV2 feature dimension
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.feature_dim = 512  # ResNet18 feature dimension
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        self.model.eval()
        self.model.to(device)
        
        if use_imagenet_norm:
            # ImageNet normalization - good for transfer learning from pretrained models
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Simpler normalization - may work better for cat-specific videos
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            ])
        
    def extract_features_from_frame_path(self, frame_path):
        """Extract CNN features from a frame file path"""
        image = Image.open(frame_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(input_tensor)
            features = features.view(features.size(0), -1)
            
        return features.cpu().numpy().flatten()
    
    def extract_features_from_clip(self, clip_path):
        """Extract features from all frames in a clip directory"""
        frame_files = sorted(glob.glob(os.path.join(clip_path, "*.jpg")))
        features = []
        
        for frame_file in frame_files:
            feature = self.extract_features_from_frame_path(frame_file)
            features.append(feature)
            
        return np.array(features)
    
class ActionSequenceDataset(Dataset):
    def __init__(self, data_root, sequence_length=5, stride=1, feature_extractor=None, action_to_label=None):
        """
        Args:
            data_root: Path to cat_video_frames directory
            sequence_length: Number of frames in each sequence (matches LSTM seq_len)
            stride: Step size for sliding window WITHIN each clip (not across clips)
            feature_extractor: CNNFeatureExtractor instance
            action_to_label: Dict mapping action names to labels {'scratching': 0, 'playing': 1, ...}
        """
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.stride = stride
        self.feature_extractor = feature_extractor
        self.action_to_label = action_to_label or {'scratching': 0}  # Default for current single action
        
        self.sequences = []
        self.labels = []
        self.clip_info = []  # Track which clip each sequence comes from
        self._prepare_sequences()
        
    def _prepare_sequences(self):
        total_sequences = 0
        
        for action_name in os.listdir(self.data_root):
            action_path = os.path.join(self.data_root, action_name)
            if not os.path.isdir(action_path):
                continue
                
            action_label = self.action_to_label.get(action_name, 0)
            
            for clip_name in os.listdir(action_path):
                clip_path = os.path.join(action_path, clip_name)
                if not os.path.isdir(clip_path):
                    continue
                    
                frame_files = sorted(glob.glob(os.path.join(clip_path, "*.jpg")))
                
                if len(frame_files) < self.sequence_length:
                    print(f"Warning: Clip {clip_name} has only {len(frame_files)} frames, skipping (need {self.sequence_length})")
                    continue
                
                # Create sliding window sequences ONLY within this clip
                clip_sequences = 0
                for start_idx in range(0, len(frame_files) - self.sequence_length + 1, self.stride):
                    sequence_frames = frame_files[start_idx:start_idx + self.sequence_length]
                    self.sequences.append(sequence_frames)
                    self.labels.append(action_label)
                    self.clip_info.append(f"{action_name}/{clip_name}")
                    clip_sequences += 1
                    
                print(f"Clip {clip_name}: {len(frame_files)} frames → {clip_sequences} sequences")
                total_sequences += clip_sequences
                    
        print(f"Total: {total_sequences} sequences from {len(self.action_to_label)} actions")
        # print(f"Sequences per clip: {[self.clip_info.count(clip) for clip in set(self.clip_info)]}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns:
            sequence_features: Tensor of shape (sequence_length, feature_dim)
            label: Action label (integer)
        """
        frame_paths = self.sequences[idx]
        label = self.labels[idx]
        
        # Extract features for each frame in sequence
        sequence_features = []
        for frame_path in frame_paths:
            features = self.feature_extractor.extract_features_from_frame_path(frame_path)
            sequence_features.append(features)
            
        sequence_features = np.array(sequence_features)
        
        return torch.FloatTensor(sequence_features), torch.LongTensor([label])
    
    def get_clip_info(self, idx):
        """Get information about which clip a sequence belongs to"""
        return self.clip_info[idx]
    
    def get_action_distribution(self):
        """Get the count of sequences per action"""
        action_counts = {}
        for label in self.labels:
            action_counts[label] = action_counts.get(label, 0) + 1
        return action_counts
    
    def create_weighted_sampler(self):
        """Create a weighted sampler for balanced training"""
        # Count sequences per action
        action_counts = self.get_action_distribution()
        
        # Print distribution
        print("\nAction distribution:")
        for action_name, action_label in self.action_to_label.items():
            count = action_counts.get(action_label, 0)
            print(f"  {action_name} (label {action_label}): {count} sequences")
        
        # Calculate weights (inverse frequency)
        weights = []
        for label in self.labels:
            # Weight = 1 / (count of this action)
            weight = 1.0 / action_counts[label]
            weights.append(weight)
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True  # Allow replacement for balanced sampling
        )
        
        return sampler
    
device = torch.device('mps:0')
print(f"Using device: {device}")

# Initialize feature extractor with ImageNet normalization option
# Try both approaches to see which works better for cat videos
print("=== Testing with image normalization ===")
feature_extractor_imagenet = CNNFeatureExtractor(
    model_name='mobilenet_v3_small', 
    device=device, 
    use_imagenet_norm=True
)

# print("=== Testing with simple normalization ===")
# feature_extractor_simple = CNNFeatureExtractor(
#     model_name='mobilenet_v3_small', 
#     device=device, 
#     use_imagenet_norm=False
# )

print(f"Feature dimension: {feature_extractor_imagenet.feature_dim}")

data_root = '../cat_video_frames'

sequence_length = 8
dataset = ActionSequenceDataset(
    data_root=data_root,
    sequence_length=sequence_length,
    stride=3,  # Every 3 frames within each clip
    feature_extractor=feature_extractor_imagenet,  
    action_to_label={'scratching': 0, 'eating': 1}
)

print(f"\nDataset size: {len(dataset)}")
if len(dataset) > 0:
    print(f"First sequence shape: {dataset[0][0].shape}")
    print(f"First sequence clip: {dataset.get_clip_info(0)}")
    print(f"Last sequence clip: {dataset.get_clip_info(-1)}")
    print(f"First label: {dataset[0][1]}")

batch_size = 8

# Create balanced dataloader using weighted sampling
balanced_sampler = dataset.create_weighted_sampler()
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=balanced_sampler)

for batch_idx, (sequences, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print(f"  Sequences shape: {sequences.shape}")  # Should be (batch_size, sequence_length, feature_dim)
    print(f"  Labels shape: {labels.shape}")       # Should be (batch_size, 1)
    print(f"  Labels: {labels.flatten()}")
    if batch_idx == 0:  
        break

