import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb

from src.pseudo_diloco.config import Config, load_config_from_yaml
from src.pseudo_diloco.models.resnet import get_resnet_18
from src.pseudo_diloco.pseudo_diloco import PseudoDiloco

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
        model: torch.nn.Module,
        config: Config,
    ):

    diloco = PseudoDiloco(
        model=model,
        M=1,
        config=config,
    )

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

    model = diloco.get_active_model().to(device)
    model = torch.compile(model)
    inner_opt = diloco.get_active_inner_optimizer()
    scheduler = diloco.get_active_scheduler()
    loss_fn = torch.nn.CrossEntropyLoss()
    wandb.init(
        project="pseudo-diloco",
        name=f"resnet-18-cifar10",
        config=config.model_dump(),
    )
    local_steps = 0
    for epoch in range(config.training_config.num_epochs):
        for step, (images, labels) in enumerate(train_dataloader):
            local_steps += 1
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = loss_fn(out, labels)
            loss.backward()

            grad_norm = torch.nn.utils.get_total_norm(model.parameters(), 2)

            inner_opt.step()
            inner_opt.zero_grad()

            dict_to_log = {
                "loss": loss.item(),
                "epoch": epoch,
                "step": step,
                "local_step": local_steps,
                "lr": [group["lr"] for group in inner_opt.param_groups][0],
                "grad_norm": grad_norm.item(),
            }
            wandb.log(dict_to_log)

            if local_steps % config.training_config.local_steps == 0:
                diloco.outer_step()
                diloco.sync_replicas()
                # With one replica this is a no-op.
                diloco.iterate_replica()
                local_steps = 0
        scheduler.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                val_loss = 0
                val_acc = 0
                for batch_idx, (images, labels) in enumerate(val_dataloader):
                    images = images.to(device)
                    labels = labels.to(device)
                    out = model(images)
                    loss = loss_fn(out, labels)
                    val_loss += loss.item()
                    _, preds = out.max(1)
                    val_acc += preds.eq(labels).sum().item()
                val_loss /= config.dataset_config.val_size
                val_acc /= config.dataset_config.val_size
                print(f"Epoch {epoch} Val Loss {val_loss} Val Acc {val_acc}")

            wandb.log({
                "epoch": epoch,
                "step": step,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
            print(f"Epoch {epoch}, Step {step}: Val loss: {val_loss}, Val acc: {val_acc}")

    with torch.no_grad():
        val_loss = 0
        val_acc = 0
        for _, (images, labels) in enumerate(val_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = loss_fn(out, labels)
            val_loss += loss.item()
            _, preds = out.max(1)
            val_acc += preds.eq(labels).sum().item()
        val_loss /= config.dataset_config.val_size
        val_acc /= config.dataset_config.val_size
        wandb.log({
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(f"Final Val loss: {val_loss}, Final Val acc: {val_acc}")

    print("Training complete")
    wandb.finish()

def sweep_train():
    wandb.init()
    
    config = load_config_from_yaml("configs/resnet-18-cifar-10-baseline.yml")
    config.outer_optimizer_config.lr = wandb.config.outer_lr
    config.outer_optimizer_config.momentum = wandb.config.outer_momentum
    
    model = get_resnet_18(
        num_classes=config.dataset_config.num_classes,
    )
    
    train(model, config)

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