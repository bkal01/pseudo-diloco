import torch
import wandb

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    get_cosine_schedule_with_warmup,
)


def main():
    # model params
    vocab_size = 32000
    hidden_size = 128
    intermediate_size = 512
    num_attention_heads = 4
    num_hidden_layers = 6
    rms_norm_eps = 1e-05
    seq_len = 1024

    # training params
    batch_size = 512
    per_device_batch_size = 32
    inner_lr = 4e-4
    weight_decay = 0.1
    b1 = 0.9
    b2 = 0.95

    outer_lr = 0.7
    momentum = 0.9
    num_warmup_steps = 1_000
    num_training_steps = 88_000
    local_steps = 500

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        rms_norm_eps=rms_norm_eps,
        use_cache=False,
    )
    model = LlamaForCausalLM(
        config=config,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="mistralai/Mistral-7B-v0.1",
        use_fast=True,
    )
    tokenizer.pad_token = "</s>"

    ds = load_dataset(
        path="NeelNanda/c4-code-20k",
        name="default",
    )
    print(f"Dataset size: {len(ds['train']):,} examples")

    def tokenize_function(data):
        outputs = tokenizer(data["text"], truncation=True, max_length=seq_len)
        return outputs

    tokenized_datasets = ds.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        collate_fn=data_collator,
        batch_size=per_device_batch_size,
    )

    inner_opt = torch.optim.AdamW(
        params=model.parameters(),
        lr=inner_lr,
        weight_decay=weight_decay,
        betas=(b1, b2),
    )
    scheduler = get_cosine_schedule_with_warmup(
        inner_opt,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    outer_opt = torch.optim.SGD(
        params=model.parameters(),
        lr=outer_lr,
        momentum=momentum,
        nesterov=True,
    )
    orig_params = [
        p.data.detach().clone().to("cpu")
        for g in outer_opt.param_groups
        for p in g["params"]
    ]

    model.train()

    loss_batch = 0

    gradient_accumulation_steps = batch_size // per_device_batch_size

    wandb.init(project="pseudo_diloco")

    # We do gradient accumulation with a micro batch size of 32, accumulating to a batch size of 512
    for micro_step, micro_batch in enumerate(train_dataloader):
        real_step = (micro_step + 1) // gradient_accumulation_steps
        step_within_grad_acc = (micro_step + 1) % gradient_accumulation_steps

        out = model(**micro_batch)
        loss = out.loss / gradient_accumulation_steps
        loss_batch += loss.detach()
        loss.backward()

        if step_within_grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            inner_opt.step()
            scheduler.step()
            inner_opt.zero_grad()

            if real_step % local_steps == 0:
                new_params = [
                    p.data.detach().clone().to("cpu")
                    for g in inner_opt.param_groups
                    for p in g["params"]
                ]
                for orig_param, new_param in zip(orig_params, new_params):
                    d_orig_param = orig_param.data.to(new_param.device)
                    new_param.grad = d_orig_param - new_param.data
                    # no all reduce here because we just have one worker
                    new_param.data = d_orig_param

                outer_opt.step()
                outer_opt.zero_grad()
                orig_params = [
                    p.data.detach().clone().to("cpu")
                    for g in outer_opt.param_groups
                    for p in g["params"]
                ]

            dict_to_log = {
                "Loss": loss_batch.item(),
                "step": real_step,
                "lr": [group["lr"] for group in inner_opt.param_groups][0],
                "Perplexity": torch.exp(loss_batch).item(),
                "total_samples": real_step * batch_size,
            }

            wandb.log(dict_to_log)
            print(
                f"step: {real_step}, loss: {loss_batch.item()}, lr {[group['lr'] for group in inner_opt.param_groups][0]}"
            )
            loss_batch = 0

    wandb.finish()
    print("Training completed.")


if __name__ == "__main__":
    main()
