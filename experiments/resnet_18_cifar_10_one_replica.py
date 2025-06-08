import wandb
from src.pseudo_diloco.config import load_config_from_yaml
from src.pseudo_diloco.models.resnet import get_resnet_18
from src.pseudo_diloco.datasets import get_cifar10_dataloaders
from src.pseudo_diloco.train import train_classification

def sweep_train():
    wandb.init()
    
    config = load_config_from_yaml("configs/resnet-18-cifar-10-one-replica.yml")
    config.outer_optimizer_config.lr = wandb.config.outer_lr
    config.outer_optimizer_config.momentum = wandb.config.outer_momentum
    
    model = get_resnet_18(
        num_classes=config.dataset_config.num_classes,
    )
    
    train_dataloader, val_dataloader, test_dataloader = get_cifar10_dataloaders(config)
    
    train_classification(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        config=config,
        wandb_name=f"resnet-18-cifar10-one-replica-lr{wandb.config.outer_lr}-mom{wandb.config.outer_momentum}"
    )

sweep_config = {
    'method': 'grid',
    'parameters': {
        'outer_lr': {
            'values': [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        'outer_momentum': {
            'values': [0.9, 0.95]
        }
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="pseudo-diloco")
    print(f"Starting sweep: {sweep_id}")
    wandb.agent(sweep_id, sweep_train)