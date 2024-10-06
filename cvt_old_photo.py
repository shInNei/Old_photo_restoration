import cv2
import numpy as np
import os
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt

def apply_sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter)
    return np.clip(sepia_image, 0, 255).astype(np.uint8)

def add_noise(image):
    noise = np.random.normal(0, 50, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image.astype(np.float32), noise.astype(np.float32))
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def reduce_saturation(image):
    faded_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    faded_image[:, :, 1] = faded_image[:, :, 1] * 0.4  # Reduce saturation
    faded_image[:, :, 2] = faded_image[:, :, 2] * 0.9  # Slightly reduce brightness
    return cv2.cvtColor(faded_image, cv2.COLOR_HSV2BGR)

def apply_blur(image):
    old_photo = cv2.GaussianBlur(image.astype(np.float32), (7, 7), 0)
    return np.clip(old_photo, 0, 255).astype(np.uint8)

def add_vignette(image):
    rows, cols = image.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, cols / 2)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows / 2)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    vignette_image = image.copy()
    
    for i in range(3):  # Apply vignette effect to each channel
        vignette_image[:, :, i] *= mask
        
    return vignette_image

def make_photo_look_old(image):
    image = apply_sepia(image)
    image = add_noise(image)
    image = reduce_saturation(image)
    return apply_blur(image)

def process_celebA_images(dataset, original_folder, old_folder, num_images=10, img_size=(512, 512)):
    # Ensure both output folders exist
    os.makedirs(original_folder, exist_ok=True)
    os.makedirs(old_folder, exist_ok=True)

    # Process a specified number of images
    for i in range(num_images):
        # Load image from dataset
        image = dataset[i]['image']

        image = image.resize(img_size)
        
        # Convert PIL Image to NumPy array
        image_np = np.array(image)

        # Convert to old photo style
        old_photo = make_photo_look_old(image_np)

        # Save the original image
        original_path = os.path.join(original_folder, f'original_photo_{i + 1}.jpg')
        image.save(original_path)  # Save the original image directly from PIL

        # Save the old-style image
        old_photo_path = os.path.join(old_folder, f'old_photo_{i + 1}.jpg')
        cv2.imwrite(old_photo_path, old_photo)

    print(f'Saved {num_images} original and old-style image pairs to {original_folder} and {old_folder}.')

# Transform for the CelebA dataset
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the images to the desired size
    transforms.ToTensor(),         # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
])

# Load the CelebA dataset
dataset = load_dataset("nielsr/CelebA-faces", split='train')

# Process and save paired images (original and old-style)
process_celebA_images(dataset, "original_photos", "old_style_photos", num_images=2000, img_size=(512, 512))

