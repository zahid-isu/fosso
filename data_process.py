import typing
from torchvision import datasets, transforms
import torch.utils.data


# Data selected for training and testing.
DataType = typing.Literal[
    "cifar",
    "fashionmnist",
    "kmnist",
    "mnist",
    "usps",
    "svhn"
]
#DATASET_CONFIG dictionary
DATASET_CONFIG = {
    "mnist": {
        "dataset_cls": datasets.MNIST,
        "transform": transforms.ToTensor(),
    },
    "kmnist": {
        "dataset_cls": datasets.KMNIST,
        "transform": transforms.ToTensor(),
    },
    "fashionmnist": {
        "dataset_cls": datasets.FashionMNIST,
        "transform": transforms.ToTensor(),
    },
    "usps": {
        "dataset_cls": datasets.USPS,
        "transform": transforms.ToTensor(),
    },
    "svhn": {
        "dataset_cls": datasets.SVHN,
        "transform": transforms.ToTensor(),
    },
    "cifar": {
        "dataset_cls": datasets.CIFAR10,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # normalize for cifer dataset
        ]),
    },
}

def processing_data(
    dataset: DataType,
    batch_size: int = 512,
    train_data_ratio: float = 1.0,
    num_workers: int = 2,
)-> torch.utils.data.DataLoader:
    """Process the input data.

    Args:
        dataset: the input dataset name string.
        batch_size: the size of batch.
        train_data_len: the length of training data taken from raw data.
            Length can be varying within the total length.
        test_data_len: the length of testing data taken from raw data.
            Length can be varying within the total length.
        cifar_norm: the normalization for cifar10 dataset.
        num_workers: the number of workers.

    Returns:
        The processed dataloaders from training and testing.
    """
    config = DATASET_CONFIG.get(dataset)
    if config is None:
        raise ValueError("Invalid dataset name")

    dataset_cls = config["dataset_cls"]
    transform = config["transform"]

    if dataset == "svhn":
        train_data = dataset_cls(
            root="data",
            split='train',
            download=True,
            transform=transform,
        )
        test_data = dataset_cls(
            root="data",
            split='test',
            download=True,
            transform=transform,
        )
    else:
        train_data = dataset_cls(
            root="data",
            train=True,
            download=True,
            transform=transform,
        )
        test_data = dataset_cls(
            root="data",
            train=False,
            download=True,
            transform=transform,
        )
    # Select a portion of data if MNIST dataset.
    
    train_data_len= int(len(train_data) * train_data_ratio)
    test_data_len= int(len(test_data) * train_data_ratio)
    train_data = list(train_data)[:train_data_len]
    test_data = list(test_data)[:test_data_len]
    train_data = [(x, y) for x, y in train_data]
    test_data = [(x, y) for x, y in test_data]
        
    # Generate dataloaders.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=500, # Set up a batch size for completeness.
        shuffle=False,
    )

    print(f"Train dataset size: {len(train_data)}")
    print(f"Test dataset size: {len(test_data)}")

    return train_loader, test_loader
