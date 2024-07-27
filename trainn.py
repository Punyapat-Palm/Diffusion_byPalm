import torch
from data import DiffSet  # Ensure you have the necessary imports for DiffSet
from model import DiffusionModel
from config import config
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

train_dataset = DiffSet(True)
val_dataset = DiffSet(False)

train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], num_workers=4, shuffle=True, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=config['batch_size'], num_workers=4, shuffle=True, persistent_workers=True
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
    # Ensure the output directory exists
    os.makedirs(opdir, exist_ok=True)

    model.to(device)
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

        # Validation
        val_loss = validate_model(model, val_loader, device)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}')

        # Save the model if the validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            last_epoch = epoch + 1
            torch.save(model.state_dict(), '{}/epochs{}.pth'.format(opdir ,config['max_epoch']))
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')

    print('Training loss: {} When epoch is: {}'.format(best_val_loss, last_epoch))

if __name__ == "__main__":
    t_range = config['diffusion_steps']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config['max_epoch']

    print('The model is training on {}'.format(device))

    # Initialize model, optimizer
    model = DiffusionModel(train_dataset.size * train_dataset.size, config["diffusion_steps"], train_dataset.depth, device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    os.makedirs('./result/Stage{}'.format(config['diagnosis']), exist_ok=True)
    num_version = max([int(str(f.name)[-1:]) for f in os.scandir('result/Stage{}'.format(config['diagnosis'])) if f.is_file() == False], default = 0) + 1
    output_dir = 'result/Stage{}/version_{}'.format(config['diagnosis'], num_version)

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, epochs, output_dir, device)

    print('The model saved at {}'.format(output_dir))
