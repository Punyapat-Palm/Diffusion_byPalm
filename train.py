from datetime import datetime
start_time = datetime.now()
import torch
from data import DiffSet
from model import DiffusionModel
from config import config
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
import os
import re
import logging

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
torch.cuda.empty_cache()

train_dataset = DiffSet(True)
val_dataset = DiffSet(False)

train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], num_workers=16, shuffle=True, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=config['batch_size'], num_workers=16, shuffle=True, persistent_workers=True
)

def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    progress_bar = tqdm(val_loader, desc='Validating', leave=False)
    with torch.no_grad():
        for data in progress_bar:
            data = data.to(device)
            loss = model.get_loss(data)
            val_loss += loss.item()
            progress_bar.set_postfix(loss=val_loss/(progress_bar.n+1))
    return val_loss / len(val_loader)

def train_model(model, train_loader, val_loader, optimizer, epochs, opdir, device):
    model.to(device)

    best_model_path = None
    best_val_loss = float('inf')
    last_epoch = int

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, data in progress_bar:
            data = data.to(device)
            optimizer.zero_grad()
            loss = model.get_loss(data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=epoch_loss/(batch_idx+1))

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_epoch_loss:.4f}')
        logging.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_epoch_loss:.4f}')

        # Validation
        val_loss = validate_model(model, val_loader, device)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.5f}')
        logging.info(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.5f}')

        # Save the model if the validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            last_epoch = epoch + 1
            new_best_model_path = '{}/epoch_{}.pth'.format(opdir, last_epoch)
            if best_model_path:
                # Remove old best model file
                os.remove(best_model_path)
            model_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'depth': config['depth'],
                'image_size': config['image_size'],
                'diffusion_steps': config['diffusion_steps']
            }
            torch.save(model_data, new_best_model_path)
            best_model_path = new_best_model_path
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')
            logging.info(f'New best model saved with validation loss: {best_val_loss:.4f}')
        else:
            print(f'Training loss: {best_val_loss:.8f} when epoch is: {last_epoch}')
            logging.info(f'Training loss: {best_val_loss:.8f} when epoch is: {last_epoch}')

        if epoch - last_epoch > 250:
            model.load_state_dict(torch.load(new_best_model_path, map_location=device, weights_only=True))
            print('Load model from the last best model')
            logging.info('Load model from the last best model')

if __name__ == "__main__":
    t_range = config['diffusion_steps']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config['max_epoch']

    # Initialize model, optimizer
    model = DiffusionModel(train_dataset.size * train_dataset.size, config["diffusion_steps"], train_dataset.depth, device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    #Create output folder
    num_version = max(
    	[int(re.findall(r'\d+', f.name)[-1]) for f in os.scandir('result/Stage{}'.format(config['diagnosis'])) if not f.is_file()],
    	default=0
    ) + 1
    output_dir = './result/Stage{}/version_{}'.format(config['diagnosis'], num_version)
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(output_dir, "outputT.log"), level=logging.INFO, format='%(asctime)s - %(message)s')

    print('The model is training on {}'.format(device))
    print('The model saved at {}'.format(output_dir))

    logging.info('The model is training on {}'.format(device))
    logging.info('The model saved at {}'.format(output_dir))

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, epochs, output_dir, device)
    print('The model saved at {}'.format(output_dir))
    logging.info('The model saved at {}'.format(output_dir))

    #check time that use for training
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    logging.info('Duration: {}'.format(end_time - start_time))
