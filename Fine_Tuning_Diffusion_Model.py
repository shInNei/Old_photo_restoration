from diffusers import AutoPipelineForImage2Image
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import Dataset
from transformers import AutoTokenizer
from diffusers.image_processor import VaeImageProcessor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate the ImageProcessor to access its config
processor = VaeImageProcessor()
# vae_latent_channels = processor.config.vae_latent_channels

# Some Parameters
num_epochs = 1
learning_rate = 1e-5
batch_size = 2

def show_images(celeba_batch, old_batch, num_images=1):
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))
    axes = axes.flatten()

    for i in range(num_images):
        # Plot CelebA images
        celeba_img = celeba_batch[i].permute(1, 2, 0).cpu().numpy()
        celeba_img = (celeba_img * 255).astype(np.uint8)  # Convert from float16/32 to uint8
        axes[2 * i].imshow(celeba_img)
        axes[2 * i].set_title(f'CelebA Image {i + 1}')
        axes[2 * i].axis('off')

        # Plot Old Photo images
        old_img = old_batch[i].permute(1, 2, 0).cpu().numpy()
        old_img = (old_img * 255).astype(np.uint8)  # Convert from float16/32 to uint8
        axes[2 * i + 1].imshow(old_img)
        axes[2 * i + 1].set_title(f'Old Photo Image {i + 1}')
        axes[2 * i + 1].axis('off')

    plt.tight_layout()
    plt.show()

# pipeline =  AutoPipelineForImage2Image.from_pretrained(
#    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
#).to(device)

# tokenizer = AutoTokenizer.from_pretrained(pipeline)
# tokenizer.clean_up_tokenization_spaces = True
pipeline = AutoPipelineForImage2Image.from_pretrained('./fine_tuned_pipeline', torch_dtype=torch.float16, variant="fp16").to(device)
# Define dataset and dataloader
dataset = Dataset.PairedImageDataset(celeba_folder="original_photos", old_photo_folder="converted_old_photos", img_size=(256, 256))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define Optimizer
optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=learning_rate, weight_decay=1e-4)

losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    num_images_processed = 0

    for celeba_batch, old_batch in tqdm(dataloader, total=len(dataloader)):
        if num_images_processed >= 25:
            break
            
        # Move batches to the correct device
        celeba_batch = celeba_batch.to(device).to(torch.float16)
        old_batch = old_batch.to(device).to(torch.float16)
        
        # Generate enhanced images using the pipeline
        with torch.no_grad():
            enhanced_images = pipeline(
                prompt=["Enhance and color correct this old photo"] * old_batch.shape[0], 
                image=old_batch
            ).images

        # Convert the enhanced images to a tensor
        enhanced_images = torch.stack(
            [torch.tensor(np.array(enhanced_img), dtype=torch.float16).permute(2, 0, 1).to(device).requires_grad_()
             for enhanced_img in enhanced_images]
        )

        # Loss function
        mse_loss = F.l1_loss(enhanced_images, celeba_batch)

        # Zero gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        mse_loss.backward()

        # Update model parameters
        optimizer.step()

        # Store loss for logging
        losses.append(mse_loss.item())

        num_images_processed += len(celeba_batch)

    # Print the average loss for the epoch
    avg_loss = sum(losses) / len(losses)
    print(f"Average loss for epoch {epoch + 1}: {avg_loss}")

# Save the fine-tuned model
pipeline.save_pretrained('./fine_tuned_pipeline')

