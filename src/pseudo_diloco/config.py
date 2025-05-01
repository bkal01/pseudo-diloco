from pydantic import BaseModel

class LMConfig(BaseModel):
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    rms_norm_eps: float
    seq_len: int

class TokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str
    use_fast: bool
    pad_token: str

class DatasetConfig(BaseModel):
    path: str
    name: str
    streaming: bool

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

class TrainingConfig(BaseModel):
    batch_size: int
    per_device_batch_size: int
    token_budget: int
    local_steps: int

class Config(BaseModel):
    lm_config: LMConfig
    inner_optimizer_config: AdamWConfig
    outer_optimizer_config: SGDConfig
    scheduler_config: CosineSchedulerConfig
