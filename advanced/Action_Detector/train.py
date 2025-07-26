
from LSTM import newLSTM  
import torch
from dataset import *
import time
from tqdm import tqdm

device = torch.device('mps:0')

# Initialize feature extractor with ImageNet normalization option
# Try both approaches to see which works better for cat videos
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
    stride=3, 
    feature_extractor=feature_extractor_imagenet,  
    action_to_label={'scratching': 0, 'eating': 1, 'playing': 2}
)

print(f"\nDataset size: {len(dataset)}")
if len(dataset) > 0:
    print(f"First sequence shape: {dataset[0][0].shape}")
    print(f"First label: {dataset[0][1]}")
batch_size = 32

balanced_sampler = dataset.create_weighted_sampler()
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=balanced_sampler)

for batch_idx, (sequences, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print(f"  Sequences shape: {sequences.shape}")  # Should be (batch_size, sequence_length, feature_dim)
    print(f"  Labels shape: {labels.shape}")       # Should be (batch_size, 1)
    print(f"  Labels: {labels.flatten()}")
    if batch_idx == 0:  
        break

input_size = feature_extractor_imagenet.feature_dim  # 576 for MobileNetV3-Small
hidden_size = 512  # Increased for better capacity
output_size = 3   # Binary classification (scratching vs non-scratching)

lstm_model = newLSTM(input_size, hidden_size, output_size)
print(f"[*] LSTM Model:")
print(f"  Input size: {input_size}")
print(f"  Hidden size: {hidden_size}")
print(f"  Output size: {output_size}")
print(f"  Sequence length: {sequence_length}")

# Move model to device
lstm_model = lstm_model.to(device)

# Training setup
import torch.optim as optim

# Split dataset into train/validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create train and validation dataloaders with optimized settings
# Note: num_workers=0 for MPS compatibility, pin_memory=False for Mac
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, pin_memory=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=False)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-4)

# Training parameters
num_epochs = 2
best_val_acc = 0.0
start_time = time.time()

print(f"\n[*] Training Setup:")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Val samples: {len(val_dataset)}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: 0.001")
print(f"  Epochs: {num_epochs}")

# Training loop
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    # Training phase
    lstm_model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    # Training progress bar
    train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                     leave=False, dynamic_ncols=True)
    
    for batch_idx, (sequences, labels) in enumerate(train_pbar):
        sequences = sequences.to(device)
        labels = labels.to(device).squeeze()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = lstm_model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        # Update progress bar
        current_acc = 100 * train_correct / train_total if train_total > 0 else 0
        train_pbar.set_postfix({
            'Loss': f'{train_loss/(batch_idx+1):.4f}',
            'Acc': f'{current_acc:.2f}%'
        })
    
    # Validation phase
    lstm_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    # Validation progress bar
    val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', 
                   leave=False, dynamic_ncols=True)
    
    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(val_pbar):
            sequences = sequences.to(device)
            labels = labels.to(device).squeeze()
            
            outputs = lstm_model(sequences)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            # Update validation progress bar
            current_val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            val_pbar.set_postfix({
                'Loss': f'{val_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_val_acc:.2f}%'
            })
    
    # Calculate metrics
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    epoch_time = time.time() - epoch_start_time
    elapsed_time = time.time() - start_time
    
    # Estimate remaining time
    avg_epoch_time = elapsed_time / (epoch + 1)
    eta_seconds = avg_epoch_time * (num_epochs - epoch - 1)
    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
    
    print(f'\nEpoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.1f}s, ETA: {eta_str}')
    print(f'  Train Loss: {train_loss/len(train_dataloader):.4f}, Train Acc: {train_acc:.2f}%')
    print(f'  Val Loss: {val_loss/len(val_dataloader):.4f}, Val Acc: {val_acc:.2f}%')
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': lstm_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc
        }, 'best_action_classifier.pth')
        print(f'  🎉 New best model saved! Val Acc: {val_acc:.2f}%')

total_time = time.time() - start_time
print(f'\n🎯 Training completed in {total_time/60:.1f} minutes!')
print(f'🏆 Best validation accuracy: {best_val_acc:.2f}%')

