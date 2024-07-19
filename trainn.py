import torch
from data import DiffSet  # Ensure you have the necessary imports for DiffSet
from model import DiffusionModel
from config import config
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import imageio
import glob
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import os

train_dataset = DiffSet(True)
val_dataset = DiffSet(False)
train_loader = DataLoader(train_dataset, batch_size=4, num_workers=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4, shuffle=True)

def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    progress_bar = tqdm(val_loader, desc='Validating', leave=False)
    with torch.no_grad():
        for data in progress_bar:
            data = data.to(device)
            loss = model.get_loss(data, 0)  # Assuming batch_idx is not used in get_loss
            val_loss += loss.item()
            progress_bar.set_postfix(loss=val_loss/(progress_bar.n+1))
    return val_loss / len(val_loader)

def train_model(model, train_loader, val_loader, optimizer, epochs, device):
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, data in progress_bar:
            data = data.to(device)
            optimizer.zero_grad()
            loss = model.get_loss(data, batch_idx)
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
            torch.save(model.state_dict(), 'epochs{}_stage{}_V{}.pth'.format(config['max_epoch'], config['diagnosis'], config['version']))
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')

if __name__ == "__main__":
    t_range = config['diffusion_steps']  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config['max_epoch']

    # Initialize model, optimizer
    model = DiffusionModel(train_dataset.size * train_dataset.size, config["diffusion_steps"], train_dataset.depth, device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, epochs, device)
    torch.save(model.state_dict(), 'epochs{}_stage{}_V{}.pth'.format(config['max_epoch'], config['diagnosis'], config['version']))
