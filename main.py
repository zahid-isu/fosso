import argparse
import copy
from data_process import processing_data, DataType
from models import ConvolutionalNeuralNet, MultilayerPerceptron, LinearModel, LogisticRegression, DenseNet,ResNet20
from optimizer import run_optimization, Optimizer, OptimizationOutput
from plotting import plotting
import torch
from tqdm import tqdm
# import wandb


# Set up argument parsing
parser = argparse.ArgumentParser(description="Train models on various datasets.")
parser.add_argument("--batch_size", type=int, default=1024, help="Input batch size for training (default: 512)")
parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train (default: 20)")
parser.add_argument("--dataset", type=str, default="cifar", choices=["mnist", "cifar", "kmnist", "fashionmnist"], help="Dataset to use (default: mnist)")
parser.add_argument("--model", type=str, default="dense_net", choices=["cnn", "mlp", "linear_model", "logistic_regression", "dense_net", "resnet20"], help="Choose the raw model to use (default: dense_net)")
parser.add_argument("--criterion", type=str, default="epoch", choices=["epoch", "loss", "gradient"], help="Criterion for FOSSO optimizer")
parser.add_argument("--optimizer", type=str, nargs="*", default=["adam", "adam+sgd", "fosso", "lbfgs", "sgd"], help="Optimizer. Default: all")

args = parser.parse_args()

# Create a dictionary to map model names to model classes
model_classes = {
    "cnn": ConvolutionalNeuralNet,
    "mlp": MultilayerPerceptron,
    "linear_model": LinearModel,
    "logistic_regression": LogisticRegression,
    "dense_net": lambda: DenseNet(growth_rate=32, num_layers=5, num_classes=10),
    "resnet20":ResNet20
}

if __name__ == "__main__":
    # wandb.init(project="fosso", entity="zahid13")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model_classes[args.model]()  # Instantiate the model based on args.model
    model = model.to(device)
    loss_dict, accuracy_dict, gradient_dict = {}, {}, {}
    optims = args.optimizer

    train_loader, test_loader = processing_data(
        batch_size=args.batch_size,
        dataset=args.dataset,
    )
    for opt in optims:
        print(f"starting training for {opt} optimizer {len(train_loader)*args.batch_size} train samples...")
        model = copy.deepcopy(model)

        # Add the optimizer to the wandb configuration
        # wandb.config.update({
        #     "optimizer": opt,
        #     "batch_size": args.batch_size,
        #     "learning_rate_sgd": 0.01,
        #     "learning_rate_adam": 0.001,
        #     "num_epochs": args.num_epochs,
        #     "model": args.model,
        #     "dataset": args.dataset
        # })
        
        loss, accuracy, gradient_norm = run_optimization(
            model=model,
            opt=opt,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate_sgd=0.01,
            learning_rate_adam=0.001,
            num_epochs=args.num_epochs,
            dataset=args.dataset,
            batch_size=args.batch_size
        )
        loss_dict[opt] = loss
        accuracy_dict[opt] = accuracy
        gradient_dict[opt] = gradient_norm
    plotting(
        optims=optims,
        loss_dict=loss_dict,
        accuracy_dict=accuracy_dict,
        gradient_dict=gradient_dict,
        dataset=args.dataset,
        model=args.model
    )
    for opt in optims:
        print(f"Accuracy for {opt}: {accuracy_dict[opt][-1]};")


## commandline: python main.py --batch_size 512 --num_epochs 10 --dataset mnist
        