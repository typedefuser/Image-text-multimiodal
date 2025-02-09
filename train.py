import torch
import torch.optim as optim
import torch.nn as nn
from models import VisionLanguageModel
from prepare_data import VLM_Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from pathlib import Path

# Hyperparameters
epochs = 10
batch_size = 8
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories for checkpoints
Path("checkpoints").mkdir(exist_ok=True)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (images, texts) in enumerate(dataloader):
        # Move data to device
        images = images.to(device)
        texts = {k: v.to(device) for k, v in texts.items()}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, texts["input_ids"][:, 0])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            images_per_sec = (batch_idx + 1) * batch_size / (time.time() - start_time)
            print(f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {avg_loss:.4f} "
                  f"Speed: {images_per_sec:.2f} images/sec")
    
    return total_loss / len(dataloader)

def main():
    # Initialize model and move to device
    print(f"Using device: {device}")
    model = VisionLanguageModel().to(device)
    
    # Load dataset
    dataset = VLM_Dataset(
        "data/images", 
        "data/annotations/captions_val2017.json",
        max_samples=1000
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        
        # Train one epoch
        train_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        
        # Update learning rate
        scheduler.step(train_loss)
        
        # Save checkpoint if best model
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            print(f"Saved new best model with loss: {train_loss:.4f}")
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"Average Loss: {train_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

if __name__ == "__main__":
    main()