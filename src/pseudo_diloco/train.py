import torch
import wandb
from .config import Config
from .pseudo_diloco import PseudoDiloco

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_classification(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    config: Config,
    wandb_name: str = "experiment"
):
    diloco = PseudoDiloco(
        model=model,
        M=config.training_config.num_replicas,
        config=config,
    )

    model = diloco.get_active_model().to(device)
    inner_opt = diloco.get_active_inner_optimizer()
    scheduler = diloco.get_active_scheduler()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    wandb.init(
        project="pseudo-diloco",
        name=wandb_name,
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
            inner_opt.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.get_total_norm(model.parameters(), 2)

            inner_opt.step()

            dict_to_log = {
                "loss": loss.item(),
                "epoch": epoch,
                "step": step,
                "local_step": local_steps,
                "active_replica": diloco.active_replica,
                "lr": [group["lr"] for group in inner_opt.param_groups][0],
                "grad_norm": grad_norm.item(),
            }
            wandb.log(dict_to_log)

            if local_steps % config.training_config.local_steps == 0:
                if diloco.active_replica == diloco.M - 1:
                    # We've iterated through all the replicas, so we need to take a step in our outer
                    # optimization loop, then sync all the replicas.
                    diloco.outer_step()
                    diloco.sync_replicas()
                    val_loss, val_acc = _evaluate(
                        model, val_dataloader, loss_fn
                    )

                    wandb.log({
                        "epoch": epoch,
                        "step": step,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    })
                    print(f"Epoch {epoch}, Step {step}: Val loss: {val_loss}, Val acc: {val_acc}")

                # Otherwise we just move the current replica to the CPU and move on to the next one.
                diloco.iterate_replica()
                model = diloco.get_active_model()
                inner_opt = diloco.get_active_inner_optimizer()
                scheduler = diloco.get_active_scheduler()
                local_steps = 0
                
        scheduler.step()

    val_loss, val_acc = _evaluate(
        model, val_dataloader, loss_fn
    )
    wandb.log({
        "val_loss": val_loss,
        "val_acc": val_acc,
    })
    print(f"Final Val loss: {val_loss}, Final Val acc: {val_acc}")

    print("Training complete")
    wandb.finish()


def _evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = loss_fn(out, labels)
            batch_size = labels.size(0)
            
            total_loss += loss.item() * batch_size
            _, preds = out.max(1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += batch_size
    
    model.train()
    return total_loss / total_samples, total_correct / total_samples 