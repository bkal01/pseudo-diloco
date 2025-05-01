import torch
import wandb

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
)

from src.pseudo_diloco.config import *
from src.pseudo_diloco.pseudo_diloco import PseudoDiloco


def train(config: Config):
    llama_config = LlamaConfig.from_dict(config.lm_config.model_dump())
    model = LlamaForCausalLM(
        config=llama_config,
    )
    diloco = PseudoDiloco(
        model=model,
        M=1,
        config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.tokenizer_config.pretrained_model_name_or_path,
        use_fast=config.tokenizer_config.use_fast,
    )
    tokenizer.pad_token = config.tokenizer_config.pad_token

    ds = load_dataset(
        path=config.dataset_config.path,
        name=config.dataset_config.name,
        streaming=config.dataset_config.streaming,
    )

    def tokenize_function(data):
        outputs = tokenizer(
            data["text"],
            truncation=True,
            max_length=config.lm_config.seq_len,
        )
        return outputs

    tokenized_datasets = ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        collate_fn=data_collator,
        batch_size=config.training_config.per_device_batch_size,
    )

    gradient_accumulation_steps = config.training_config.batch_size // config.training_config.per_device_batch_size


    model = diloco.get_active_model().to("cuda")
    inner_opt = diloco.get_active_inner_optimizer()
    scheduler = diloco.get_active_scheduler()
    for micro_step, micro_batch in enumerate(train_dataloader):
        real_step = (micro_step + 1) // gradient_accumulation_steps
        step_within_grad_acc = (micro_step + 1) % gradient_accumulation_steps

        for key in micro_batch.keys():
            micro_batch[key] = micro_batch[key].to("cuda")

        out = model(**micro_batch)
        loss = out.loss / gradient_accumulation_steps
        loss_batch += loss.detach()
        loss.backward()

        if step_within_grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            inner_opt.step()
            scheduler.step()
            inner_opt.zero_grad()

            if real_step % config.training_config.local_steps == 0:
                if active_replica == diloco.M - 1:
                    diloco.outer_step()
                    diloco.sync_replicas()
                    active_replica = 0
                else:
                    diloco.replicas[active_replica].to("cpu")
                    active_replica += 1
                
                model = diloco.replicas[active_replica].to("cuda")
                inner_opt = diloco.inner_optimizers[active_replica]
                scheduler = diloco.schedulers[active_replica]
        
        


if __name__ == "__main__":
    config = Config(
        lm_config=LMConfig(
            vocab_size=32000,
            hidden_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            intermediate_size=512,
            rms_norm_eps=1e-05,
        ),
        inner_optimizer_config=AdamWConfig(
            lr=4e-4,
            weight_decay=0.1,
            b1=0.9,
            b2=0.95,
        ),
        outer_optimizer_config=SGDConfig(
            lr=0.7,
            momentum=0.9,
            nesterov=True,
        ),
        scheduler_config=CosineSchedulerConfig(
            num_warmup_steps=1000,
            num_training_steps=88000,
        ),
    )

    train(config)