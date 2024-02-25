import argparse
import copy
from data_process import processing_data, DataType
from models import ConvolutionalNeuralNet, MultilayerPerceptron, LinearModel, LogisticRegression, DenseNet3,ResNet20
from optimizer import run_optimization, Optimizer, OptimizationOutput
from plotting import plotting, plot_loss_landscape
import pandas as pd
import torch
import os
from tqdm import tqdm
import numpy as np
# import wandb


# Set up argument parsing
parser = argparse.ArgumentParser(description="Train models on various datasets.")
parser.add_argument("--batch_size", type=int, default=512, help="Input batch size for training (default: 512)")
parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train (default: 20)")
parser.add_argument("--dataset", type=str, default="cifar", choices=["mnist", "cifar", "kmnist", "fashionmnist", "usps", "svhn"], help="Dataset to use (default: mnist)")
parser.add_argument("--model", type=str, default="dense_net", choices=["cnn", "mlp", "linear_model", "logistic_regression", "dense_net", "resnet20"], help="Choose the raw model to use (default: dense_net)")
parser.add_argument("--criterion", type=str, default="epoch", choices=["epoch", "loss", "gradient"], help="Criterion for FOSSO optimizer")
parser.add_argument("--optimizer", type=str, nargs="*", default=["adam", "adam+sgd", "fosso", "lbfgs", "sgd"], help="Optimizer. Default: all")
parser.add_argument("--num_layer", type=int, default=5, help="Number of layers in DenseNet")
parser.add_argument("--train_ratio", type=float, default=1.0, help="ratio of training data to use (default: 1.0)")

seeds =[0, 1, 2, 42]
args = parser.parse_args()

# Create a dictionary to map model names to model classes
model_classes = {
    "cnn": ConvolutionalNeuralNet,
    "mlp": MultilayerPerceptron,
    "linear_model": LinearModel,
    "logistic_regression": LogisticRegression,
    "dense_net": lambda: DenseNet3(args.num_layer, num_classes=10, growth_rate=32,
                                   reduction=0.5, bottleneck=False, dropRate=0.0),
    "resnet20":ResNet20
}

if __name__ == "__main__":
    # wandb.init(project="fosso", entity="zahid13")

    if torch.cuda.is_available():
        device_id = 0  
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize the untrained model 
    original_model_class = model_classes[args.model]
    original_model = original_model_class() if not callable(original_model_class) else original_model_class()
    

    loss_dict, accuracy_dict, gradient_dict = {}, {}, {}
    optims = args.optimizer

    train_loader, test_loader = processing_data(
        batch_size=args.batch_size,
        dataset=args.dataset,
        train_data_ratio=args.train_ratio
    )
    acc = {opt: [] for opt in args.optimizer}

    for seed in seeds:
        torch.manual_seed(seed)
        for opt in optims:
            print(f"{original_model} layer={args.num_layer}starting training for {opt} optimizer {len(train_loader)*args.batch_size} train samples...")
            model = copy.deepcopy(original_model)
            model = model.to(device)

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
            model=args.model,
            batch_size= args.batch_size,
            num_epoch=args.num_epochs,
            num_layer=args.num_layer,
            train_data_ratio=args.train_ratio
        )

        # plot_loss_landscape(model=model,
        #                     model_name=args.model,
        #                     train_loader=train_loader)

        # collect the final accuracies for each optim
        for opt in optims:
            acc[opt].append(accuracy_dict[opt][-1])
            print(f"For seed {seed} final acc {opt}:", accuracy_dict[opt][-1])
    
    #create the csv to store results
    results = []
    for opt in optims:
        mean_acc = np.mean(acc[opt])
        var_acc = np.var(acc[opt])
        results.append([opt, mean_acc, var_acc])
        print("Results: ", results)
    
    # df = pd.DataFrame(results, columns=["Optimizer", "Mean Accuracy", "Variance"])
    # csv_path = os.path.join("results", f"{args.model}_optimizer_acc_{args.dataset}_{args.train_ratio}.csv")
    # os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # df.to_csv(csv_path, index=False)

    # print("Results saved to:", csv_path)



## commandline: python main.py --batch_size 512 --num_epochs 10 --dataset mnist
        