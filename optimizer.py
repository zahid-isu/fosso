import typing
import torch
import torch.optim as optim
import torch.nn as nn
from models import ModelType
from data_process import DataType
from utils import calculate_gradient_norm, get_accuracy
from models import ConvolutionalNeuralNet, MultilayerPerceptron, LinearModel, LogisticRegression
from tqdm import tqdm


# Optimizer type selected.
Optimizer = typing.Literal["adam","adam+sgd","fosso","lbfgs","sgd"]

# Criterion for FOSSO optimizer.
Criterion = typing.Literal["epoch","loss","gradient"]

# Optimization output.
OptimizationOutput = typing.Tuple[
    typing.List[float],
    typing.List[float],
    typing.List[float],
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_optimization(
    model: ModelType,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    opt: Optimizer = "fosso",
    learning_rate_sgd: float = 0.01,
    learning_rate_adam: float = 0.001,
    num_epochs: int = 10,
    history_size: int = 10,
    max_eval: int = 4,
    line_search_fn: str = "strong_wolfe",
    criterion: Criterion = "epoch",
    dataset: DataType = "mnist",
    batch_size: int = 512,
) -> OptimizationOutput:
    """Run the optimization.

    Args:
        model: the input model.
        train_loader: input training data.
        test_loader: testing data.
        opt: the optimizer.
        learning_rate_sgd: the learning rate for SGD.
        learning_rate_adam: the learning rate for Adam.
        num_epochs: the number of epochs.
        history_size: the history size for L-BFGS.
        max_eval: the maximum number of evaluation.
        line_search_fn: the line search function.
        criterion: the criterion applied to FOSSO.
        datatype: the type of input data.
        batch_size: the size of batch.

    Returns:
        A tuple including loss, accuracy and gradient norm values.
    """
    # Define the loss function.
    criterion = nn.CrossEntropyLoss()
    # training
    model.train()
    iters, losses, test_acc = [], [], []
    gradient_norm = []
    
    # Define optimizer objects
    optimizers = {
    "adam": optim.Adam(model.parameters(), lr=learning_rate_adam),
    "sgd": optim.SGD(model.parameters(), lr=learning_rate_sgd),
    "lbfgs": optim.LBFGS(model.parameters(), history_size=10, max_eval=4, line_search_fn="strong_wolfe")
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0  # total epoch loss
        num_batches = 0 
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", total=len(train_loader))
        for xs, ts in iter(train_loader_tqdm):
            print(f"Processing batch {num_batches+1}")
            xs, ts = xs.to(device), ts.to(device) 
            if opt in ["adam", "sgd", "lbfgs"]:
                optimizer = optimizers[opt]
            elif opt == "fosso":
                if not criterion == "epoch":
                    if epoch < 3:
                        # Need some warm up by Adam.
                         optimizer = optimizers["adam"]
                    else:
                        if criterion == "gradient":
                            if total_norm <= 0.5: # TODO: move to argument
                                optimizer = optimizers["lbfgs"]
                            else:
                                optimizer = optimizers["adam"]
                        else:
                            if loss < 2.: # TODO: move to argument
                                optimizer = optimizers["lbfgs"]
                            else:
                                optimizer = optimizers["adam"]
                else:
                    if epoch >= 5: # switching epoch 
                        optimizer = optimizers["lbfgs"]
                    else:
                        optimizer = optimizers["adam"]
            elif opt == "adam+sgd":
                if epoch >= num_epochs / 2:
                    optimizer = optimizers["sgd"]
                else:
                    optimizer = optimizers["adam"]
            if len(ts) != batch_size:
                continue
            if not dataset == "cifar":
                if isinstance(model, ConvolutionalNeuralNet):
                    s = xs.view(-1, 1, 28, 28)
                else:
                    # Flatten the image.
                    xs = xs.view(-1, 784)
            if isinstance(optimizer, optim.LBFGS):
                def closure():
                    optimizer.zero_grad()
                    zs = model(xs)
                    loss = criterion(zs, ts)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                loss = closure()
            else:
                optimizer.zero_grad()    # Clean up step for PyTorch.
                zs = model(xs)
                loss = criterion(zs, ts) # Compute the total loss.
                loss.backward()          # Compute updates for each parameter.
                optimizer.step()         # Make the updates for each parameter.
            # Save the current training information.
            # Tconverted to average epoch loss. Will reduce curve variance.
            total_loss += float(loss.item()) # sum the loss for current epoch
            num_batches += 1
            train_loader_tqdm.set_postfix(loss=loss.item())
            # losses.append(float(loss)/batch_size)  # avg batch loss
        # Get gradient norm, test acc and avg_epoch_loss
        average_epoch_loss = total_loss / num_batches 
        losses.append(average_epoch_loss)
        total_norm = calculate_gradient_norm(model.parameters())
        gradient_norm.append(total_norm)
        test_acc.append(get_accuracy(model, test_loader, dataset, device)) 
        print(f"Epoch {epoch + 1} average epoch_loss: {average_epoch_loss}") 
        train_loader_tqdm.close()
    out = (losses, test_acc, gradient_norm)
    return out