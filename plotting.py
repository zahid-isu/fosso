import copy
import loss_landscapes
import loss_landscapes.metrics
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
import numpy as np
from datetime import datetime
import typing
import torch
from models import ModelType
import os


def plot_loss_landscape(model: ModelType, 
                        model_name: typing.List[str],
                        train_loader: torch.utils.data.DataLoader,
                        save_path: str = "./3d_fig",
                        steps=40
):
    """
    Plot the loss landscape of the model.
    
    Args:
        model: The trained model.
        train_loader: DataLoader for the training dataset.
        steps: Number of steps for the loss landscape plot resolution.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"creating 3D loss-landscape of {model_name}...")
    model_dir = os.path.join(save_path, f"{model_name}")
    os.makedirs(model_dir, exist_ok=True)

    model_final = copy.deepcopy(model)
    criterion = torch.nn.CrossEntropyLoss()
    x, y = next(iter(train_loader))
    
    metric = loss_landscapes.metrics.Loss(criterion, x, y)
    loss_data = loss_landscapes.random_plane(model_final.cpu(), metric, 10, steps, normalization='filter', deepcopy_model=True)
    fig = plt.figure(figsize=(8,8), dpi=300)
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(steps)] for i in range(steps)])
    Y = np.array([[i for _ in range(steps)] for i in range(steps)])
    
    ax.plot_surface(X, Y, loss_data, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title(f'3D Loss Landscape of model{model_name}')

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"3D_loss_landscape_{current_time}_{model_name}_plot.png"))
    plt.close(fig)


def plotting(
    optims: typing.List[str],
    loss_dict: typing.Dict[str, typing.List[float]],
    accuracy_dict: typing.Dict[str, typing.List[float]],
    gradient_dict: typing.Dict[str, typing.List[float]],
    dataset: typing.List[str],
    model: typing.List[str],
    batch_size: typing.List[str],
    num_epoch: typing.List[str],
    num_layer: typing.List[str],
    train_data_ratio: typing.List[float],
    save_path: str = "./fig" # Define plots path
):
    """Plot the loss, accuracy, gradient norm.

    Args:
        optims: the optimizers.
        loss_dict: dictionary consisting losses for optimizers.
        accuracy_dict: dictionary consisting accuracies for optimizers.
        gradient_dict: dictionary consisting gradient norm for optimizers.
        dataset: dataset name
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Ensure the save path exists, create if it doesn't
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Define plot characteristics in a list
    plot_specs = [
        ('Training Loss', 'Epochs', 'Loss', loss_dict),
        ('Testing Accuracy', 'Epochs', 'Accuracy', accuracy_dict),
        ('Gradient Norm', 'Epochs', 'Gradient Norm', gradient_dict)
    ]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), dpi=300)
    model_dir = os.path.join(save_path, f"{model}")
    os.makedirs(model_dir, exist_ok=True)


    # Loop over each subplot specification
    for ax, (title, xlabel, ylabel, data_dict) in zip(axes, plot_specs):
        ax.set_title(title)
        for opt in optims:
            epochs = np.arange(1, len(data_dict[opt]) + 1) 
            ax.plot(epochs, data_dict[opt], label=f'{opt}', linewidth=2.5)
        ax.tick_params(axis='x', labelsize=14)  
        ax.tick_params(axis='y', labelsize=14) 
        ax.set_xlabel(xlabel, fontsize=14) 
        ax.set_ylabel(ylabel, fontsize=14)  
        ax.set_title(title, fontsize=16)
        ax.legend(loc='best', fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # fig.suptitle(f"{dataset}_{model}_batch{batch_size}_epoch{num_epoch}_lay{num_layer}", fontsize=18, y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"{current_time}_{dataset}_ratio{train_data_ratio}_{model}_batch{batch_size}_plot.png"))
    plt.close(fig)
