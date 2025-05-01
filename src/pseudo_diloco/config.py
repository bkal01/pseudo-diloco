from pydantic import BaseModel

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

class Config(BaseModel):
    inner_optimizer_config: AdamWConfig
    outer_optimizer_config: SGDConfig
    scheduler_config: CosineSchedulerConfig
