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



device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Some Parameters
num_epochs = 2
learning_rate = 1e-5
batch_size = 1

pipeline =  AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to(device)

# tokenizer = AutoTokenizer.from_pretrained(pipeline)
# tokenizer.clean_up_tokenization_spaces = True

# Define dataset and dataloader
celeba_dataset = Dataset.CelebADataset(split='train', folder_path = "original_photos", img_size=(512, 512))
old_dataset = Dataset.OldPhotoDataset(split='train', folder_path="converted_old_photos", img_size=(512, 512))

celeba_dataloader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True)
old_dataloader = DataLoader(old_dataset, batch_size=batch_size, shuffle=True)

# Define Optimizer
optimizer = torch.optim.SGD(pipeline.unet.parameters(), lr=learning_rate)

losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for celeba_batch, old_batch in tqdm(zip(celeba_dataloader, old_dataloader), total=len(celeba_dataloader)):
        # Move batches to the correct device
        celeba_batch = celeba_batch.to(device)
        old_batch = old_batch.to(device)

        enhanced_images = []
        
        for img in old_batch:
            # Generate enhanced images using the pipeline
            with torch.no_grad():
                enhanced_img = pipeline(prompt="Enhance this old photo", image=img).images[0]

            enhanced_img = np.array(enhanced_img)
            
            # Convert the enhanced images to a tensor if necessary
            enhanced_img = torch.tensor(enhanced_img, dtype=torch.float16).permute(2, 0, 1).to(device).requires_grad_()

            enhanced_images.append(enhanced_img)
        
        enhanced_images = torch.stack(enhanced_images).to(device)
        
        # Loss function
        l1_loss = F.l1_loss(enhanced_images, celeba_batch)

        # Zero gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        l1_loss.backward()

        # Update model parameters
        optimizer.step()

        # Store loss for logging
        losses.append(l1_loss.item())

    # Print the average loss for the epoch
    avg_loss = sum(losses) / len(losses)
    print(f"Average loss for epoch {epoch+1}: {avg_loss}")

# Save the fine-tuned model
pipeline.save_pretrained('./fine_tuned_pipeline')

