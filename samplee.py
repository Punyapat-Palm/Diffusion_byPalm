import torch
from model import DiffusionModel  # Import your DiffusionModel class
from data import DiffSet  # Import if needed
from PIL import Image
import numpy as np
from config import config
import math
import os
from tqdm.auto import tqdm


def sample_images(model, output_dir, sample_batch_size=1, sample_steps=1000, device='cuda'):
    model = model.to(device)
    model.eval()

    # Generate random noise
    x = torch.randn(
        (sample_batch_size, train_dataset.depth, model.in_size, model.in_size),
        device=device
    )

    intermediate_images = []

    # Denoise the initial noise for T steps
    for t in tqdm(range(sample_steps), desc="Sampling"):
        x = model.denoise_sample(x, model, t)
        
        # Save intermediate images periodically
        if t % 50 == 0:
            intermediate_sample = (x.clamp(-1, 1) + 1) / 2
            intermediate_sample = (intermediate_sample * 255).type(torch.uint8)
            intermediate_sample = intermediate_sample.permute(0, 2, 3, 1).cpu().numpy()
            intermediate_images.append(intermediate_sample[0]) 

    # Get the final image after denoising
    final_sample = (x.clamp(-1, 1) + 1) / 2
    final_sample = (final_sample * 255).type(torch.uint8)
    final_sample = final_sample.permute(0, 2, 3, 1).cpu().numpy()  # Convert to HWC format

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save individual final images
    for i in range(final_sample.shape[0]):
        img = Image.fromarray(final_sample[i])
        img.save(os.path.join(output_dir, f"final_sample_{i:02d}.png"))

    # Create a grid of images
    grid_size = math.ceil(sample_batch_size**.5)
    img_size = final_sample.shape[1]
    grid_img = Image.new('RGB', (img_size * grid_size, img_size * grid_size))

    for i in range(sample_batch_size):
        img = Image.fromarray(final_sample[i])
        grid_x = (i % grid_size) * img_size
        grid_y = (i // grid_size) * img_size
        grid_img.paste(img, (grid_x, grid_y))

    grid_img.save(os.path.join(output_dir, "final_grid.png"))
    print(f'The image is saved in {output_dir}')

    # Save GIF of intermediate images
    gif_images = [Image.fromarray(img) for img in intermediate_images]
    gif_path = os.path.join(output_dir, "sampling_process.gif")
    gif_images[0].save(
        gif_path,
        save_all=True,
        append_images=gif_images[1:],
        duration=100,
        loop=0
    )

if __name__ == "__main__":
    # Path to your diffusion_model.pth
    model_checkpoint = 'epochs{}_stage{}_V{}.pth'.format(config['max_epoch'], config['diagnosis'], config['version'])
    output_directory = '/sampled_images'
    train_dataset = DiffSet(True)

    # Initialize your DiffusionModel sure to initialize with appropriate parameters
    t_range = config['diffusion_steps']  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config['max_epoch']

    # Initialize model, optimizer
    model = DiffusionModel(train_dataset.size * train_dataset.size, config["diffusion_steps"], train_dataset.depth, device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))

    # Sample images using the model
    sample_images(model, output_directory, sample_batch_size=4, sample_steps=1000, device=device)
