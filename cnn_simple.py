# --------- Normalization Constants ---------
IMG_SIZE = 256
RADIUS_MIN, RADIUS_MAX = 3, 20

# --------- Dataset Classes ---------
class CircleOnTheFlyDataset(Dataset):
    def __init__(self, n_samples=10000, noise_level=2):
        self.n_samples = n_samples
        self.noise_level = noise_level

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img, label = generate_circle(noise_level=self.noise_level)
        img = np.expand_dims(img, axis=0)  # shape: (1, 256, 256)
        # Normalize targets
        row = label['row'] / IMG_SIZE
        col = label['col'] / IMG_SIZE
        radius = (label['radius'] - RADIUS_MIN) / (RADIUS_MAX - RADIUS_MIN)
        target = torch.tensor([row, col, radius], dtype=torch.float32)
        return torch.tensor(img, dtype=torch.float32), target

class CircleDataset(Dataset):
    def __init__(self, data_path):
        self.files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.json')]
        # Filter out potentially disrupted files
        self.files = self._filter_disrupted_files(self.files) 
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        with open(self.files[idx], 'r') as f:
            data = json.load(f)
        img = np.array(data['img'], dtype=np.float32)
        label = data['label']
        img = np.expand_dims(img, axis=0)  # Add channel dimension
        
        # Apply same normalization as the on-the-fly dataset
        row = label['row'] / IMG_SIZE
        col = label['col'] / IMG_SIZE
        radius = (label['radius'] - RADIUS_MIN) / (RADIUS_MAX - RADIUS_MIN)
        
        target = torch.tensor([row, col, radius], dtype=torch.float32)
        return torch.tensor(img, dtype=torch.float32), target
        
    def _filter_disrupted_files(self, files):
        """
        Filters out potentially disrupted JSON files by checking their content.
        """
        filtered_files = []
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    json.load(f)  # Attempt to load the JSON data
                filtered_files.append(file_path)  # Add to filtered list if successful
            except json.JSONDecodeError:
                print(f"Skipping disrupted file: {file_path}")
        return filtered_files

# ... (rest of your code)

# --------- CNN Model Definition ---------
class CircleFinderCNN(nn.Module):
    def __init__(self):
        super(CircleFinderCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Add an additional pooling layer to reduce feature map size
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Feature map size calculation:
        # Input: 256x256 -> conv1 -> 256x256 -> pool -> 128x128
        # -> conv2 -> 128x128 -> pool2 -> 64x64
        # So we have 32 channels of 64x64 feature maps
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 256x256 -> 128x128
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # 128x128 -> 64x64
        x = x.view(-1, 32 * 64 * 64)  # Flatten the feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# --------- Training Loop ---------
def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-4, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    def custom_loss(pred, target):
        pos_loss = F.mse_loss(pred[:, 0:2], target[:, 0:2])
        radius_loss = F.mse_loss(pred[:, 2], target[:, 2])
        return pos_loss + 2 * radius_loss

    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = custom_loss(outputs, targets)
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
                loss = custom_loss(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)  # Update learning rate based on validation loss

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"New best model! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # Load best model before evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Plot loss
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Custom Weighted Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model

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

# --------- Example Usage ---------
if __name__ == '__main__':
    # Create datasets - on-the-fly for training, fixed for validation
    train_dataset = CircleOnTheFlyDataset(n_samples=8000)
    val_dataset = CircleDataset(data_path='drive/MyDrive/data')  # Using data folder

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Print dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CircleFinderCNN()
    
    # Apply weight initialization
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(weights_init)
    
    # Train with slightly higher learning rate
    model = train_model(model, train_loader, val_loader, device, epochs=100, lr=5e-4, patience=15)

    # Evaluate on validation set
    show_predictions(model, val_dataset, device, num_samples=5)