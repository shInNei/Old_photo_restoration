import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PairedImageDataset(Dataset):
    def __init__(self, celeba_folder="original_photos", old_photo_folder="converted_old_photos", img_size=(64,64)):
        """
        Initializes a dataset that loads paired images from two folders.
        The order of images in the two folders is assumed to be corresponding.
        
        Args:
            folder_path_celeba (str): Path to the folder containing CelebA images.
            folder_path_old (str): Path to the folder containing old photos.
            img_size (tuple): Size to which images will be resized.
        """
        self.celeba_folder = celeba_folder
        self.old_photo_folder = old_photo_folder
        self.img_size = img_size

        # Ensure both folders contain the same number of images and matching filenames
        self.celeba_images = sorted([os.path.join(celeba_folder, file) for file in os.listdir(celeba_folder) if file.endswith(".jpg")])
        self.old_images = sorted([os.path.join(old_photo_folder, file) for file in os.listdir(old_photo_folder) if file.endswith(".jpg")])

        assert len(self.celeba_images) == len(self.old_images), "The two folders must contain the same number of images."

        # Define transformations: Resize, Convert to Tensor, and Random Horizontal Flip
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.celeba_images)

    def __getitem__(self, idx):
        # Get corresponding image paths
        celeba_img_path = self.celeba_images[idx]
        old_img_path = self.old_images[idx]
        
        # Load the images
        celeba_image = Image.open(celeba_img_path).convert("RGB")
        old_image = Image.open(old_img_path).convert("RGB")
        
        # Apply transformations
        celeba_image = self.transform(celeba_image)
        old_image = self.transform(old_image)

        return celeba_image, old_image




