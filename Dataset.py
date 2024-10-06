import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

class CelebADataset(Dataset):
    def __init__(self, split = 'train', folder_path = "original_photos", img_size=(64,64)):
        """
        Initializes the Old photo dataset

        Args:
            
        
        """
        self.folder_path = folder_path
        self.img_size = img_size

        self.dataset = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".jpg")]

        # Define transformation: Resize and Convert to Tensor
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
        ])
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset[idx]
        
        # Load the image
        image = Image.open(img_path).convert("RGB")  # Ensure the image is in RGB format
        
        # Apply transformations
        image = self.transform(image)

        return image

class OldPhotoDataset(Dataset):
    def __init__(self, split = 'train', folder_path = "converted_old_photos", img_size=(64,64)):
        """
        Initializes the Old photo dataset

        Args:
            
        
        """
        self.folder_path = folder_path
        self.img_size = img_size

        self.dataset = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".jpg")]

        # Define transformation: Resize and Convert to Tensor
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
        ])
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = self.dataset[idx]
        
        # Load the image
        image = Image.open(img_path).convert("RGB")  # Ensure the image is in RGB format
        
        # Apply transformations
        image = self.transform(image)

        return image



