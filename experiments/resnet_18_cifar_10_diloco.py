import wandb
from src.pseudo_diloco.config import load_config_from_yaml
from src.pseudo_diloco.models.resnet import get_resnet_18
from src.pseudo_diloco.datasets import get_cifar10_dataloaders
from src.pseudo_diloco.train import train_classification
from src.pseudo_diloco.utils import set_seed

set_seed()

def sweep_train():
    wandb.init()
    
    config = load_config_from_yaml("configs/resnet-18-cifar-10-diloco.yml")
    config.training_config.local_steps = wandb.config.local_steps
    config.training_config.num_replicas = wandb.config.num_replicas
    
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
        wandb_name=f"resnet-18-cifar10-diloco-num-replicas-{wandb.config.num_replicas}"
    )

sweep_config = {
    'method': 'grid',
    'parameters': {
        'local_steps': {
            'values': [10, 100, 250, 500, 1000]
        },
        'num_replicas': {
            'values': [1, 2, 4, 8]
        }
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="pseudo-diloco")
    print(f"Starting sweep: {sweep_id}")
    wandb.agent(sweep_id, sweep_train)