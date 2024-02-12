import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
import numpy as np
from datetime import datetime
import typing
import os


def plotting(
    optims: typing.List[str],
    loss_dict: typing.Dict[str, typing.List[float]],
    accuracy_dict: typing.Dict[str, typing.List[float]],
    gradient_dict: typing.Dict[str, typing.List[float]],
    dataset: typing.List[str],
    model: typing.List[str],
    batch_size: typing.List[str],
    num_epoch: typing.List[str],
    save_path: str = "./plots" # Define plots path
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

    # Loop over each subplot specification
    for ax, (title, xlabel, ylabel, data_dict) in zip(axes, plot_specs):
        ax.set_title(title)
        for opt in optims:
            epochs = np.arange(1, len(data_dict[opt]) + 1) 
            ax.plot(epochs, data_dict[opt], label=f'{opt}')
        ax.tick_params(axis='x', labelsize=14)  
        ax.tick_params(axis='y', labelsize=14) 
        ax.set_xlabel(xlabel, fontsize=14) 
        ax.set_ylabel(ylabel, fontsize=14)  
        ax.set_title(title, fontsize=16)
        ax.legend(loc='best', fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))



    fig.suptitle(f"{dataset}_{model}_batch{batch_size}_epoch{num_epoch}_lay5", fontsize=16, y=0.99)
    plt.tight_layout(pad=3.0, h_pad=1.0, w_pad=1.0)
    plt.savefig(os.path.join(save_path, f"{current_time}_{dataset}_{model}_lay5_ep{num_epoch}_batch{batch_size}_plot.png"))
    plt.close(fig)

