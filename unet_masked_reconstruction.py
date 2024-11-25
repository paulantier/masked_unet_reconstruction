import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from tqdm import tqdm
import random
import torch.nn.functional as F

# Dataset definition (from your provided code)
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith('.JPEG') or fname.endswith('jpg')]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure RGB mode
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),           # Resize the shorter side to 256
    transforms.CenterCrop(224),       # Crop the image to 224x224
    transforms.ToTensor(),            # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Unet Autoencoder 
class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # Downsample
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # Downsample
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2, 2),  # Downsample
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout(0.3)
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # Upsample
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Upsample
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Upsample
            nn.ReLU(inplace=True),
        )

        self.dec0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # Output layer
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)  # 224x224
        enc2_out = self.enc2(enc1_out)  # 112x112
        enc3_out = self.enc3(enc2_out)  # 56x56

        # Bottleneck
        bottleneck_out = self.bottleneck(enc3_out)  # 28x28

        # Decoder with skip connections
        dec3_out = self.dec3(bottleneck_out) + enc3_out  # 56x56
        dec2_out = self.dec2(dec3_out) + enc2_out  # 112x112
        dec1_out = self.dec1(dec2_out) + enc1_out  # 224x224
        dec_out = self.dec0(dec1_out)

        return dec_out

# Prepare data
data_dir = './random_imagenet'
train_dataset = CustomImageDataset(root=data_dir + "/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = CustomImageDataset(root=data_dir + "/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Model, loss, optimizer
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, criterion, optimizer
model = UNetAutoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Paths for saving and loading the model
model_save_path = "unet_autoenc_checkpoints/unet_autoencoder.pth"

# Function to save the model
def save_model(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)
    print(f"Model saved to {path}")

# Function to load the model
def load_model(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Model loaded from {path}, starting from epoch {start_epoch + 1}")
    return model, optimizer, start_epoch



def add_random_black_squares(image_batch, num_squares=1, square_size=60):

    images = image_batch.clone()

    batch_size, channel, height, width = images.shape

    posx=[]
    posy=[]

    for i in range(batch_size):
        for _ in range(num_squares):
            # Random position for the top-left corner of the square
            x = random.randint(0, width - square_size)
            y = random.randint(0, height - square_size)
            
            images[i, :, y:y+square_size, x:x+square_size] = 0 
            posx.append(x)
            posy.append(y)
            
    return images, posx, posy

alpha = 1.0  # Hyperparameter for the square loss weight
beta = 1.0   # Hyperparameter for the full image loss weight
square_size = 40

def custom_loss_function(outputs, targets, posx, posy, square_size, alpha, beta):
    # Compute MSE for the whole image
    full_image_loss = F.mse_loss(outputs, targets)

    # Compute MSE for the masked square region
    square_loss = 0.0
    batch_size = targets.size(0)
    for i in range(batch_size):
        x, y = posx[i], posy[i]
        original_square = targets[i, :, y:y+square_size, x:x+square_size]
        predicted_square = outputs[i, :, y:y+square_size, x:x+square_size]
        square_loss += F.mse_loss(predicted_square, original_square)

    # Average the square loss over the batch
    square_loss /= batch_size

    # Combine losses
    total_loss = alpha * square_loss + beta * full_image_loss
    return total_loss



start_epoch = 0  # Start from scratch if no model is loaded
#load model ?
model, optimizer, start_epoch = load_model(model, optimizer, model_save_path, device)

# Training loop
max_epoch = 200



for epoch in range(start_epoch, max_epoch):
    model.train()
    train_loss = 0
    for images, _ in tqdm(train_loader):
        images = images.to(device)
        images_masked, posx, posy = add_random_black_squares(images, num_squares = 1, square_size = square_size)
        
        # Forward pass
        outputs = model(images_masked)
        loss = custom_loss_function(outputs, images, posx, posy, square_size=square_size, alpha=alpha, beta=beta)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{max_epoch}], Loss: {train_loss/len(train_loader):.4f}")
    
    # Save the model at the end of each epoch
    save_model(model, optimizer, epoch, model_save_path)

    model.eval()
    with torch.no_grad():
        for i, (images, _) in tqdm(enumerate(test_loader)):
            images = images.to(device)
            images_masked, _, _ = add_random_black_squares(images, num_squares = 1, square_size = 40)
            # Forward pass
            outputs = model(images_masked)
            
            # Save some reconstructed images
            if i == 0:  # Save the first batch
                outputs = outputs.cpu()
                #outputs = outputs * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # De-normalize
                save_image(outputs, f'unet_autoenc_checkpoints/reconstructed_images_epoch{epoch+1}.png')
                save_image(images_masked.cpu(), f'unet_autoenc_checkpoints/masked_images_epoch{epoch+1}.png')
                save_image(images.cpu(), f'unet_autoenc_checkpoints/expected_images_epoch{epoch+1}.png')
                break
