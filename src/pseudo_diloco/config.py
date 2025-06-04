from pydantic import BaseModel, Field
from typing import Literal
import yaml

class DatasetConfig(BaseModel):
    train_size: int
    val_size: int
    test_size: int | None = None
    img_size: int

class AdamWConfig(BaseModel):
    lr: float
    weight_decay: float
    b1: float
    b2: float

class SGDConfig(BaseModel):
    lr: float
    momentum: float
    nesterov: bool

class CosineSchedulerConfig(BaseModel):
    num_warmup_steps: int
    num_training_steps: int

class ResnetConfig(BaseModel):
    num_classes: int

class LlamaConfig(BaseModel):
    model_name: str
    gradient_accumulation_steps: int

class TrainingConfig(BaseModel):
    model_type: Literal["resnet", "llama"]
    per_replica_batch_size: int
    local_steps: int
    num_epochs: int | None = None
    token_budget: int | None = None
    gradient_accumulation_steps: int | None = None

class Config(BaseModel):
    architecture_config: ResnetConfig | LlamaConfig
    inner_optimizer_config: AdamWConfig
    outer_optimizer_config: SGDConfig
    scheduler_config: CosineSchedulerConfig
    training_config: TrainingConfig
    dataset_config: DatasetConfig

def load_config_from_yaml(yaml_path: str) -> Config:
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config.model_validate(config_dict)