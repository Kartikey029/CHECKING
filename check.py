# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:15:50 2025

@author: RogStrix
"""

import numpy as np 
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(args):
    wandb.init(entity=args.wandb_entity, project=args.wandb_project)

    input_size = 10
    hidden_size = 20
    output_size = 1
    num_samples = 1000
    batch_size = 32
    epochs = 10
    learning_rate = 0.001

    X = torch.randn(num_samples, input_size)
    y = torch.randn(num_samples, output_size)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNet(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item()})
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple neural network.")
    parser.add_argument("--wandb_entity", type=str, required=True, help="Wandb entity name.")
    parser.add_argument("--wandb_project", type=str, required=True, help="Wandb project name.")
    args = parser.parse_args()
    train(args)