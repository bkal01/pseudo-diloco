import wandb
from src.pseudo_diloco.config import load_config_from_yaml, Config
from src.pseudo_diloco.models.resnet import resnet20
from src.pseudo_diloco.train_resnet import train

def train_with_config():
    wandb.init()
    config = load_config_from_yaml("configs/resnet-18-cifar-10.yml")
    
    config.training_config.num_epochs = wandb.config.epochs
    config.training_config.per_replica_batch_size = wandb.config.per_replica_batch_size
    config.outer_optimizer_config.lr = wandb.config.outer_lr
    config.inner_optimizer_config.weight_decay = wandb.config.inner_weight_decay
    config.inner_optimizer_config.lr = wandb.config.inner_lr
    
    model = resnet20()
    
    train(model, config)

if __name__ == "__main__":
    sweep_config = {
        "method": "grid",
        "metric": {
            "name": "val_acc",
            "goal": "maximize"
        },
        "parameters": {
            "epochs": {
                "values": [100]
            },
            "per_replica_batch_size": {
                "values": [64, 128, 256, 512]
            },
            "inner_lr": {
                "values": [0.001, 0.01, 0.1]
            },
            "inner_weight_decay": {
                "values": [0.0, 0.001, 0.01]
            },
            "outer_lr": {
                "values": [0.1, 0.4, 0.7]
            }
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="pseudo-diloco")
    wandb.agent(sweep_id, function=train_with_config, count=18)
