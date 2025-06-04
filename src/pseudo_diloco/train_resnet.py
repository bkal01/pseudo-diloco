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

    full_train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [config.dataset_config.train_size, config.dataset_config.val_size]
    )

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

    model = diloco.get_active_model().to(device)
    inner_opt = diloco.get_active_inner_optimizer()
    scheduler = diloco.get_active_scheduler()
    loss_fn = torch.nn.CrossEntropyLoss()
    active_replica = 0
    wandb.init(
        project="pseudo-diloco",
        name=f"resnet-18-cifar10",
        config=config.model_dump(),
    )
    local_step_count = 0
    for epoch in range(config.training_config.num_epochs):
        for step, (images, labels) in enumerate(train_dataloader):
            local_step_count += 1
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = loss_fn(out, labels)
            loss.backward()

            grad_norm = torch.nn.utils.get_total_norm(model.parameters(), 2)

            inner_opt.step()
            scheduler.step()
            inner_opt.zero_grad()

            if local_step_count % config.training_config.local_steps == 0:
                local_step_count = 0
                if active_replica == diloco.M - 1:
                    diloco.outer_step()
                    diloco.sync_replicas()
                    active_replica = 0
                else:
                    diloco.replicas[active_replica].to("cpu")
                    active_replica += 1

                model = diloco.replicas[active_replica].to(device)
                inner_opt = diloco.inner_optimizers[active_replica]
                scheduler = diloco.schedulers[active_replica]

                with torch.no_grad():
                    val_loss = 0.0
                    val_acc = 0.0
                    for val_images, val_labels in val_dataloader:
                        val_images = val_images.to(device)
                        val_labels = val_labels.to(device)
                        val_out = model(val_images)
                        val_loss += loss_fn(val_out, val_labels).item()
                        val_acc += (val_out.argmax(dim=1) == val_labels).float().mean()
                    val_loss /= len(val_dataloader)
                    val_acc /= len(val_dataloader)

                    wandb.log({
                        "epoch": epoch,
                        "step": step,
                        "local_step": local_step_count,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    })
                    print(f"Epoch {epoch}, Step {step}: Val loss: {val_loss}, Val acc: {val_acc}")

            dict_to_log = {
                "loss": loss.item(),
                "epoch": epoch,
                "step": step,
                "local_step": local_step_count,
                "lr": [group["lr"] for group in inner_opt.param_groups][0],
                "grad_norm": grad_norm.item(),
            }

            wandb.log(dict_to_log)

    with torch.no_grad():
        val_loss = 0.0
        val_acc = 0.0
        for val_images, val_labels in val_dataloader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            val_out = model(val_images)
            val_loss += loss_fn(val_out, val_labels).item()
            val_acc += (val_out.argmax(dim=1) == val_labels).float().mean()
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)

        wandb.log({
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(f"Final Val loss: {val_loss}, Final Val acc: {val_acc}")

    print("Training complete")
    wandb.finish()

if __name__ == "__main__":
    config = load_config_from_yaml("configs/resnet-18-cifar-10.yml")
    model = get_resnet_18(
        num_classes=config.architecture_config.num_classes,
        img_size=config.dataset_config.img_size,
    )
    train(model, config)