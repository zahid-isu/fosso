import torch
from models import ModelType
from data_process import DataType
from models import ConvolutionalNeuralNet, MultilayerPerceptron, LinearModel, LogisticRegression



def get_accuracy(
    model: ModelType,
    dataloader: torch.utils.data.DataLoader,
    dataset: DataType = "cifar",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> float:
    """Calculate accuracy.

    Args:
        model: the trained model.
        data: the input dataset.

    Returns:
        Classification accuracy.
    """
    model.eval()
    correct, total = 0, 0
    for xs, ts in dataloader:
        if not dataset == "cifar":
            if isinstance(model, ConvolutionalNeuralNet):
                xs = xs.view(-1, 1, 28, 28)
            else:
                xs = xs.view(-1, 784) # flatten the image
        xs, ts = xs.to(device), ts.to(device)
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(ts.view_as(pred)).sum().item()
        total += int(ts.shape[0])
        return correct / total

def get_avg_epoch_loss(
    model: ModelType,
    criterion,
    dataloader: torch.utils.data.DataLoader,
    optimizer_type: str,
    opt: dict,
    epoch: int,
    batch_size: int,
) -> float:
    """Calculate average epoch loss.

    Args:
        model: The trained model.
        criterion: The loss function.
        dataloader: The DataLoader for the dataset.
        optimizer_type: The type of optimizer used ('adam', 'sgd', etc.).
        opt: Dictionary containing optimizer instances.
        epoch: Current epoch number.
        batch_size: The size of each data batch.

    Returns:
        Average epoch loss.
    """
    total_loss = 0.0
    num_batches = 0
    for xs, ts in dataloader:
        optimizer = opt[optimizer_type]
        if len(ts) != batch_size:
            continue
        # Flatten the image for certain datasets and model types
        if not dataset == "cifar":
            if isinstance(model, ConvolutionalNeuralNet):
                xs = xs.view(-1, 1, 28, 28)
            else:
                xs = xs.view(-1, 784)

        # Compute loss
        optimizer.zero_grad()
        zs = model(xs)
        loss = criterion(zs, ts)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches if num_batches > 0 else 0
    return average_loss


def calculate_gradient_norm(params) -> float: # TODO: need type hint for `params`.
    """Callculate the gradient norm.

    Returns:
        The gradient norm.
    """
    total_norm = 0
    for p in params:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
    