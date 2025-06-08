import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .config import Config

def get_cifar10_dataloaders(config: Config):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=False,
        transform=val_transform
    )

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=val_transform
    )
    
    full_size = len(train_dataset)
    train_size = config.dataset_config.train_size
    val_size = config.dataset_config.val_size
    
    indices = torch.randperm(full_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training_config.per_replica_batch_size,
        shuffle=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training_config.per_replica_batch_size * 2,
        shuffle=False,
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.training_config.per_replica_batch_size * 2,
        shuffle=False,
    )
    
    return train_dataloader, val_dataloader, test_dataloader 