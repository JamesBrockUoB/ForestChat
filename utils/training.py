import os

import torch


def save_checkpoint(state, path="model_weights"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, path="model_weights", optimizer=None, scheduler=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Loaded checkpoint from {path}")
    return checkpoint


def save_model(model, path="model_weights", filename="model_weights.pth"):
    """
    Save PyTorch model weights.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): Directory to save the model.
        filename (str): Name of the saved weights file.
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    torch.save(model.state_dict(), full_path)
    print(f"Model saved to: {full_path}")


def load_model(model, path="model_weights", filename="model_weights.pth", map_location=None):
    """
    Load PyTorch model weights into a model.

    Args:
        model (torch.nn.Module): Model instance to load weights into.
        path (str): Directory containing the weights file.
        filename (str): Name of the weights file.
        map_location (str or torch.device, optional): Device to map model to.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    full_path = os.path.join(path, filename)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"No file found at {full_path}")
    
    model.load_state_dict(torch.load(full_path, map_location=map_location))
    print(f"Model loaded from: {full_path}")
    return model
