import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from skimage.transform import rotate
from shapely.geometry import Point
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --------- Constants ---------
IMG_SIZE = 256
RADIUS_MIN, RADIUS_MAX = 3, 20

# --------- On-the-Fly Dataset with Augmentation ---------
class CircleOnTheFlyDataset(Dataset):
    def __init__(self, n_samples=15000, noise_level=2):
        self.n_samples = n_samples
        self.noise_level = noise_level

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img, label = generate_circle(noise_level=self.noise_level)
        
        # Temporarily disable all augmentations
        '''
        # Simple augmentation (slight rotation - circles are rotation invariant)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            img = rotate(img, angle, preserve_range=True)
            
        # Simple contrast adjustment
        if np.random.random() > 0.5:
            contrast = np.random.uniform(0.9, 1.1)
            img = np.clip(img * contrast, 0, None)
        '''
            
        img = np.expand_dims(img, axis=0)  # shape: (1, 256, 256)
        row = label['row'] / IMG_SIZE
        col = label['col'] / IMG_SIZE
        radius = (label['radius'] - RADIUS_MIN) / (RADIUS_MAX - RADIUS_MIN)
        target = torch.tensor([row, col, radius], dtype=torch.float32)
        return torch.tensor(img, dtype=torch.float32), target

# --------- CNN Model ---------
class SimpleCircleFinderCNN(nn.Module):
    def __init__(self):
        super(SimpleCircleFinderCNN, self).__init__()
        # First layer - simple architecture that worked before
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        
        # Fully connected layers - simpler structure
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()  # Keep sigmoid for [0,1] output range

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# --------- Loss Function ---------
def simple_circle_loss(pred, target):
    """
    Simplified MSE loss with balanced weights for position and radius
    """
    # Position loss (x,y coordinates)
    pos_loss = F.mse_loss(pred[:, 0:2], target[:, 0:2])
    
    # Radius loss (normalized)
    radius_loss = F.mse_loss(pred[:, 2], target[:, 2])
    
    # Combined loss (2:1 weight ratio for position:radius)
    return pos_loss + 2.0 * radius_loss

# --------- Weight Initialization ---------
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# --------- Data Debugging Function ---------
def debug_data(dataset, num_samples=3):
    for i in range(num_samples):
        img, target = dataset[i]
        print(f"Image shape: {img.shape}, range: [{img.min():.2f}, {img.max():.2f}]")
        print(f"Target: {target}")
        
        row = target[0] * IMG_SIZE
        col = target[1] * IMG_SIZE
        radius = target[2] * (RADIUS_MAX - RADIUS_MIN) + RADIUS_MIN
        print(f"Denormalized: row={row:.1f}, col={col:.1f}, radius={radius:.1f}")
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img.squeeze(), cmap='gray')
        circle = plt.Circle((col, row), radius, color='r', fill=False)
        plt.gca().add_patch(circle)
        plt.axis('off')
        plt.title(f"Target circle: ({col:.1f}, {row:.1f}), r={radius:.1f}")
        plt.show()

# --------- Evaluation Functions ---------
def intersection_over_union(circ1_dict, circ2_dict):
    shape1 = Point(circ1_dict['row'], circ1_dict['col']).buffer(circ1_dict['radius'])
    shape2 = Point(circ2_dict['row'], circ2_dict['col']).buffer(circ2_dict['radius'])
    return shape1.intersection(shape2).area / shape1.union(shape2).area

def evaluate_model(model, dataloader, device, iou_threshold=0.7):
    model.eval()
    matches = 0
    total = 0
    total_iou = 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs).cpu().numpy()
            targets = targets.numpy()
            for pred, true in zip(outputs, targets):
                pred_dict = {
                    'row': float(pred[0] * IMG_SIZE),
                    'col': float(pred[1] * IMG_SIZE),
                    'radius': float(pred[2] * (RADIUS_MAX - RADIUS_MIN) + RADIUS_MIN)
                }
                true_dict = {
                    'row': float(true[0] * IMG_SIZE),
                    'col': float(true[1] * IMG_SIZE),
                    'radius': float(true[2] * (RADIUS_MAX - RADIUS_MIN) + RADIUS_MIN)
                }
                iou = intersection_over_union(pred_dict, true_dict)
                total_iou += iou
                if iou > iou_threshold:
                    matches += 1
                total += 1
    
    accuracy = matches / total
    avg_iou = total_iou / total
    print(f"IoU > {iou_threshold} Accuracy: {accuracy:.2%}, Average IoU: {avg_iou:.4f}")
    return accuracy, avg_iou

# --------- Visualization Function ---------
def show_predictions(model, dataset, device, num_samples=5):
    model.eval()
    fig, axs = plt.subplots(1, num_samples, figsize=(20, 4))
    
    with torch.no_grad():
        for i in range(num_samples):
            img, true = dataset[i]
            img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device
            pred = model(img).cpu().numpy()[0]  # Get only the first (and only) item in batch
            true = true.numpy()
            
            # Calculate circle parameters and explicitly convert to Python floats
            pred_row = float(pred[0] * IMG_SIZE)
            pred_col = float(pred[1] * IMG_SIZE)
            pred_radius = float(pred[2] * (RADIUS_MAX - RADIUS_MIN) + RADIUS_MIN)
            
            true_row = float(true[0] * IMG_SIZE)
            true_col = float(true[1] * IMG_SIZE)
            true_radius = float(true[2] * (RADIUS_MAX - RADIUS_MIN) + RADIUS_MIN)
            
            axs[i].imshow(img.squeeze().cpu(), cmap='gray')
            axs[i].add_patch(plt.Circle(
                (true_col, true_row),  # (col, row)
                true_radius,
                color='lime', fill=False, linewidth=2, label='True'))
            axs[i].add_patch(plt.Circle(
                (pred_col, pred_row),
                pred_radius,
                color='red', fill=False, linewidth=2, linestyle='--', label='Pred'))
            
            # Create dictionaries with explicit Python float types
            pred_dict = {'row': pred_row, 'col': pred_col, 'radius': pred_radius}
            true_dict = {'row': true_row, 'col': true_col, 'radius': true_radius}
            iou = intersection_over_union(pred_dict, true_dict)
            
            axs[i].set_title(f"IoU: {iou:.2f}")
            axs[i].axis('off')
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()

# --------- Training Function ---------
def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    criterion = simple_circle_loss  # Use the simpler loss function
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            evaluate_model(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"New best model saved! Validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model based on validation loss")

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model

# --------- Main Execution ---------
if __name__ == '__main__':
    # Import necessary function from your data generation module
    #from create_dataset import generate_circle
    
    # Create datasets with larger samples
    train_dataset = CircleOnTheFlyDataset(n_samples=20000, noise_level=2)
    val_dataset = CircleOnTheFlyDataset(n_samples=5000, noise_level=2)
    
    # Debug data before training
    print("Debugging dataset samples:")
    debug_data(train_dataset, num_samples=2)
    
    # Create dataloaders with larger batch size for GPU efficiency
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Set device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model with proper weight initialization
    model = SimpleCircleFinderCNN()  # Using the simpler model
    model.apply(weights_init)  # Apply weight initialization
    model.to(device)
    
    # Train model with higher learning rate
    model = train_model(model, train_loader, val_loader, device, 
                       epochs=100, lr=1e-3, patience=15)  # Higher LR, fewer epochs
    
    # Final evaluation
    print("\nFinal model evaluation:")
    acc, avg_iou = evaluate_model(model, val_loader, device)
    
    # Evaluate with different IoU thresholds
    print("\nEvaluation with different IoU thresholds:")
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        print(f"IoU threshold: {threshold}")
        evaluate_model(model, val_loader, device, iou_threshold=threshold)
    
    # Show sample predictions
    show_predictions(model, val_dataset, device, num_samples=5)
    
    # Save model if accuracy is good
    # if acc > 0.7:
    #     torch.save(model.state_dict(), 'circle_detector_model.pth')
    #     print("Model saved to 'circle_detector_model.pth'")