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
train_loader = DataLoader(
    train_dataset, batch_size=4, num_workers=4, shuffle=False, persistent_workers=True  
)

def train_model(model, train_loader, optimizer, epochs, device):
    model.to(device)
    model.train()

    for epoch in range(epochs):
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
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

if __name__ == "__main__":
    t_range = config['diffusion_steps']  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config['max_epoch']

    # Initialize model, optimizer
    model = DiffusionModel(train_dataset.size * train_dataset.size, config["diffusion_steps"], train_dataset.depth, device)

    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    # Train the model
    train_model(model, train_loader, optimizer, epochs, device)
    torch.save(model.state_dict(), 'diffusion_model.pth')