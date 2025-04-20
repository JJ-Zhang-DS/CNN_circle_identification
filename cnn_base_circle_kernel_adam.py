import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from shapely.geometry import Point
#from create_dataset import generate_circle

# --------- Normalization Constants ---------
IMG_SIZE = 256
RADIUS_MIN, RADIUS_MAX = 3, 20

# --------- On-the-Fly Dataset Definition ---------
class CircleOnTheFlyDataset(Dataset):
    def __init__(self, n_samples=10000, noise_level=2):
        self.n_samples = n_samples
        self.noise_level = noise_level

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img, label = generate_circle(noise_level=self.noise_level)
        img = np.expand_dims(img, axis=0)  # shape: (1, 256, 256)
        row = label['row'] / IMG_SIZE
        col = label['col'] / IMG_SIZE
        radius = (label['radius'] - RADIUS_MIN) / (RADIUS_MAX - RADIUS_MIN)
        target = torch.tensor([row, col, radius], dtype=torch.float32)
        return torch.tensor(img, dtype=torch.float32), target

# Add circle kernel creation function
def create_circle_kernel(radius, size=None):
    """Create a kernel that responds to circles of a specific radius"""
    if size is None:
        size = 2 * radius + 3  # Instead of +5
    
    kernel = np.zeros((size, size))
    center = size // 2
    
    y, x = np.ogrid[-center:size-center, -center:size-center]
    # Create a ring with thickness of 1 pixel
    mask = (x**2 + y**2 <= (radius+0.5)**2) & (x**2 + y**2 >= (radius-0.5)**2)
    kernel[mask] = 1.0
    
    # Normalize kernel
    return kernel / kernel.sum() if kernel.sum() > 0 else kernel

# --------- CNN Model Definition with Circle Kernels ---------
class CircleFinderCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(CircleFinderCNN, self).__init__()
        
        # Standard convolutional path
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(12)
        
        # Circle-specialized kernels
        self.circle_kernels = nn.ModuleList()
        for radius in [5, 10, 15, 20]:
            kernel_size = min(2 * radius + 3, 21)
            kernel = torch.tensor(create_circle_kernel(radius, size=kernel_size), dtype=torch.float32)
            conv = nn.Conv2d(1, 1, kernel_size=kernel.shape[0], padding=kernel.shape[0]//2)
            kernel = kernel * 0.1
            conv.weight.data = kernel.unsqueeze(0).unsqueeze(0)
            self.circle_kernels.append(conv)
        
        # BatchNorm for circle features
        self.bn_circle = nn.BatchNorm2d(4)  # 4 circle filters
        
        # More convolutional layers with BN
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Dropout
        self.dropout_conv = nn.Dropout2d(dropout_rate/2)
        self.dropout_fc = nn.Dropout(dropout_rate)
        
        # FC layers with BN
        self.fc1 = nn.Linear(64 * 32 * 32, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Downsample first for circle kernels
        x_downsampled = self.pool(x)
        
        # Circle kernel path with BN
        circle_features = [F.relu(kernel(x_downsampled)) for kernel in self.circle_kernels]
        circle_features = torch.cat(circle_features, dim=1)
        circle_features = self.bn_circle(circle_features)
        circle_features = self.dropout_conv(circle_features)
        
        # Standard path with BN
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.pool(x1)
        x1 = self.dropout_conv(x1)
        
        # Combine paths
        x = torch.cat([x1, circle_features], dim=1)
        
        # Continue with BN and dropout
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # FC layers with BN and dropout
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc(x)
        
        x = self.sigmoid(self.fc3(x))
        return x

# --------- Training Loop ---------
def train_model(model, train_loader, val_loader, device, epochs=200, lr=1e-4, patience=20):
    # MUCH LOWER learning rate for circle kernels model
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,  # Changed from 1e-2 to 1e-4
        weight_decay=1e-5
    )
    
    # Add gradient clipping to prevent explosion
    max_grad_norm = 1.0  # Maximum gradient norm
    
    # Learning rate scheduler for Adam
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.2,
        patience=10,
        verbose=True,
        min_lr=1e-6
    )
    
    def loss_function(pred, target):
        return combined_loss(pred, target, alpha=0.6)

    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    # Set up the live plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Combined IoU/MSE Loss')
    ax.set_title('Loss Curve (Live Update)')
    ax.grid(True)
    train_line, = ax.plot([], [], 'b-', label='Train Loss')
    val_line, = ax.plot([], [], 'r-', label='Val Loss')
    ax.legend()
    plt.tight_layout()
    
    # Lists to store epoch numbers and learning rates
    epochs_list = []
    lr_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_function(outputs, targets)
            loss.backward()
            
            # Add gradient clipping before optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Safety check for NaN loss
            if torch.isnan(loss):
                print("Warning: NaN loss detected. Skipping batch.")
                optimizer.zero_grad()  # Clear the bad gradients
                continue

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        epochs_list.append(epoch + 1)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)  # For ReduceLROnPlateau
        
        # Store current learning rate
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        # Only print loss every 10 epochs (or for the first epoch)
        if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Update the live plot
        train_line.set_data(epochs_list, train_losses)
        val_line.set_data(epochs_list, val_losses)
        ax.relim()  # Recalculate limits
        ax.autoscale_view()  # Autoscale
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Turn off interactive mode after training
    plt.ioff()
    
    # Final static plot - now with two subplots
    plt.figure(figsize=(15, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs_list, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Combined IoU/MSE Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # Learning rate plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, lr_history, 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.show()


# --------- Evaluation and Visualization ---------
def show_predictions(model, dataset, device, num_samples=5):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, idx in enumerate(indices):
        img, true = dataset[idx]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device)).cpu().numpy()[0]
        # Denormalize
        true_row = true[0].item() * IMG_SIZE
        true_col = true[1].item() * IMG_SIZE
        true_rad = true[2].item() * (RADIUS_MAX - RADIUS_MIN) + RADIUS_MIN
        pred_row = pred[0] * IMG_SIZE
        pred_col = pred[1] * IMG_SIZE
        pred_rad = pred[2] * (RADIUS_MAX - RADIUS_MIN) + RADIUS_MIN
        axs[i].imshow(img.squeeze(), cmap='gray')
        axs[i].add_patch(plt.Circle((true_col, true_row), true_rad, color='lime', fill=False, linewidth=2, label='True'))
        axs[i].add_patch(plt.Circle((pred_col, pred_row), pred_rad, color='red', fill=False, linewidth=2, linestyle='--', label='Pred'))
        axs[i].set_title(f"IoU: {intersection_over_union({'row': true_row, 'col': true_col, 'radius': true_rad}, {'row': pred_row, 'col': pred_col, 'radius': pred_rad}):.2f}")
        axs[i].axis('off')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()

def intersection_over_union(circ1_dict, circ2_dict):
    shape1 = Point(circ1_dict['row'], circ1_dict['col']).buffer(circ1_dict['radius'])
    shape2 = Point(circ2_dict['row'], circ2_dict['col']).buffer(circ2_dict['radius'])
    return shape1.intersection(shape2).area / shape1.union(shape2).area

def evaluate_model(model, dataloader, device, iou_threshold=0.7):
    model.eval()
    matches = 0
    total = 0
    all_ious = []  # Track all IoU values
    
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs).cpu().numpy()
            targets = targets.numpy()
            for pred, true in zip(outputs, targets):
                # Denormalize for IoU calculation
                pred_row = pred[0] * IMG_SIZE
                pred_col = pred[1] * IMG_SIZE  
                pred_radius = pred[2] * (RADIUS_MAX - RADIUS_MIN) + RADIUS_MIN
                
                true_row = true[0] * IMG_SIZE
                true_col = true[1] * IMG_SIZE
                true_radius = true[2] * (RADIUS_MAX - RADIUS_MIN) + RADIUS_MIN
                
                pred_dict = {'row': pred_row, 'col': pred_col, 'radius': pred_radius}
                true_dict = {'row': true_row, 'col': true_col, 'radius': true_radius}
                
                iou = intersection_over_union(pred_dict, true_dict)
                all_ious.append(iou)
                
                if iou > iou_threshold:
                    matches += 1
                total += 1
    
    # Calculate more informative metrics
    accuracy = matches / total
    mean_iou = sum(all_ious) / len(all_ious)
    median_iou = sorted(all_ious)[len(all_ious)//2]
    
    # Show distribution in IoU brackets
    brackets = [0.5, 0.6, 0.7, 0.8, 0.9]
    bracket_counts = [sum(1 for iou in all_ious if iou > thresh) / total for thresh in brackets]
    
    print(f"IoU > {iou_threshold} Accuracy: {accuracy:.2%}")
    print(f"Mean IoU: {mean_iou:.4f}, Median IoU: {median_iou:.4f}")
    print("IoU Distribution:")
    for thresh, count in zip(brackets, bracket_counts):
        print(f"  IoU > {thresh:.1f}: {count:.2%}")
    
    return accuracy, mean_iou, all_ious

def iou_loss(pred, target, epsilon=1e-6, target_threshold=0.7):
    """Enhanced IoU loss with threshold focus."""
    # Denormalize predictions and targets
    pred_row = pred[:, 0] * IMG_SIZE
    pred_col = pred[:, 1] * IMG_SIZE
    pred_radius = pred[:, 2] * (RADIUS_MAX - RADIUS_MIN) + RADIUS_MIN
    
    target_row = target[:, 0] * IMG_SIZE
    target_col = target[:, 1] * IMG_SIZE
    target_radius = target[:, 2] * (RADIUS_MAX - RADIUS_MIN) + RADIUS_MIN
    
    # Ensure positive radius (prevent div by zero)
    pred_radius = torch.clamp(pred_radius, min=0.1)
    
    # Distance between centers
    center_dist = torch.sqrt((pred_row - target_row)**2 + (pred_col - target_col)**2)
    
    # Sum of radii
    radii_sum = pred_radius + target_radius
    
    # Larger epsilon to prevent division by very small numbers
    epsilon = 1e-4
    
    # Base IoU component calculation
    dist_loss = center_dist / (radii_sum + epsilon)
    radius_loss = torch.abs(pred_radius - target_radius) / (radii_sum + epsilon)
    base_iou_loss = dist_loss + radius_loss
    
    # Estimated IoU - approximate conversion from our distance metrics
    # This is a rough approximation to help with scaling
    est_iou = 1.0 - torch.clamp(base_iou_loss, 0.0, 1.0)
    
    # Apply increased penalty for predictions near but below threshold
    # This creates a "soft hinge" that pushes predictions to exceed the threshold
    threshold_penalty = torch.where(
        (est_iou < target_threshold) & (est_iou > target_threshold - 0.2),
        1.5 * (target_threshold - est_iou),  # 1.5x penalty in the critical region
        torch.zeros_like(est_iou)
    )
    
    # Combined loss with threshold incentive
    return base_iou_loss.mean() + threshold_penalty.mean()

def combined_loss(pred, target, alpha=0.6):
    """Combined MSE and enhanced IoU loss."""
    mse_component = F.mse_loss(pred, target)
    iou_component = iou_loss(pred, target, target_threshold=0.7)
    
    # Slightly higher alpha - focus more on IoU now that training is stable
    return alpha * iou_component + (1-alpha) * mse_component

# --------- Example Usage ---------
if __name__ == '__main__':
    train_dataset = CircleOnTheFlyDataset(n_samples=10000)
    val_dataset = CircleOnTheFlyDataset(n_samples=2000)

    # train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=100)

    # A100-optimized DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,  # Important for faster CPU->GPU data transfer
    #    persistent_workers=True  # Keeps workers alive between iterations
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=256, 
        shuffle=False,  # No need to shuffle validation data
        num_workers=4,
        pin_memory=True
    )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CircleFinderCNN()
    train_model(model, train_loader, val_loader, device, epochs=150, patience=25, lr=1e-4)
    evaluate_model(model, val_loader, device)
    show_predictions(model, val_dataset, device, num_samples=5)

    # Or with custom dropout rate
    model = CircleFinderCNN(dropout_rate=0)  # Higher dropout for more regularization

    # Much smaller kernels that still detect circles
    for radius in [5, 10, 15, 20]:
        kernel_size = min(2 * radius + 3, 21)  # Cap at 21x21
        kernel = torch.tensor(create_circle_kernel(radius, size=kernel_size), dtype=torch.float32)