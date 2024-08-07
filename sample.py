from datetime import datetime
start_time = datetime.now()
import torch
from model import DiffusionModel
from PIL import Image
import numpy as np
from config import config
import math
import glob
import re
import os
from tqdm.auto import tqdm
import logging

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
torch.cuda.empty_cache()

def sample_images(model, output_dir, sample_batch_size=1, device='cuda'):
    model = model.to(device)
    model.eval()

    # Generate random noise
    x = torch.randn(
        (sample_batch_size, model_path['depth'], model_path['image_size'], model_path['image_size']),
        device=device
    )
    intermediate_images = []
    sample_steps = torch.arange(model.t_range - 1, 0, -1)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Denoise the initial noise for T steps
    for t in tqdm(sample_steps, desc="Sampling"):
        x = model.denoise_sample(x, t)
        
        if t % 25 == 0 or t == sample_steps[-1]:
            intermediate_sample = (x.clamp(-1, 1) + 1) / 2
            intermediate_sample = (intermediate_sample * 255).type(torch.uint8)
            intermediate_sample = intermediate_sample.permute(0, 2, 3, 1).cpu().numpy()

            # Create a grid of images
            grid_size = math.ceil(sample_batch_size ** 0.5)
            img_size = intermediate_sample.shape[1]
            grid_img = Image.new('RGB', (img_size * grid_size, img_size * grid_size))

            for i in range(sample_batch_size):
                img = Image.fromarray(intermediate_sample[i])
                grid_x = (i % grid_size) * img_size
                grid_y = (i // grid_size) * img_size
                grid_img.paste(img, (grid_x, grid_y))

            intermediate_images.append(grid_img)

    # Get the final image after denoising
    final_sample = (x.clamp(-1, 1) + 1) / 2
    final_sample = (final_sample * 255).type(torch.uint8)
    final_sample = final_sample.permute(0, 2, 3, 1).cpu().numpy()  # Convert to HWC format

    # Save individual final images
    for i in range(final_sample.shape[0]):
        img = Image.fromarray(final_sample[i])
        img.save(os.path.join(output_dir, f"final_sample_{i:02d}.png"))

    # Create a grid of images for the final step
    grid_img = Image.new('RGB', (img_size * grid_size, img_size * grid_size))

    for i in range(sample_batch_size):
        img = Image.fromarray(final_sample[i])
        grid_x = (i % grid_size) * img_size
        grid_y = (i // grid_size) * img_size
        grid_img.paste(img, (grid_x, grid_y))

    grid_img.save(os.path.join(output_dir, "final_grid.png"))
    print(f'The image is saved in {output_dir}')
    logging.info(f'The image is saved in {output_dir}')

    # Save GIF of the entire grid of intermediate images
    gif_path = os.path.join(output_dir, "sampling_process_grid.gif")
    intermediate_images[0].save(
        gif_path,
        save_all=True,
        append_images=intermediate_images[1:],
        duration=100,
        loop=0
    )

if __name__ == "__main__":
    # Path to your ***.pth
    opdir = 'result/Stage{}/version_{}'.format(config['diagnosis'], config['version'])
    model_checkpoint = glob.glob(os.path.join(opdir, "*.pth"))[-1]
    num_img_fd = max(
    	[int(re.findall(r'\d+', f.name)[-1]) for f in os.scandir(opdir) if not f.is_file()],
    	default=0
    ) + 1
    output_directory = '{}/Image_{}'.format(opdir, num_img_fd)
    
    logging.basicConfig(filename=os.path.join(output_directory, "outputS.txt"), level=logging.INFO, format='%(asctime)s - %(message)s')

    # Initialize your DiffusionModel sure to initialize with appropriate parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Your Image is sampling on {}'.format(device))
    print('The model was loading from {}'.format(model_checkpoint))
    logging.info('Your Image is sampling on {}'.format(device))
    logging.info('The model was loading from {}'.format(model_checkpoint))

    # Initialize model, optimizer
    model_path = torch.load(model_checkpoint, map_location=device, weights_only=True)
    model = DiffusionModel(model_path['image_size'] * model_path['image_size'], model_path['diffusion_steps'], device)
    model.load_state_dict(model_path['model_state_dict'])

    # Sample images using the model
    sample_images(model, output_directory, sample_batch_size=config['num_sample'], device=device)

    #check time that use for training
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    logging.info('Duration: {}'.format(end_time - start_time))