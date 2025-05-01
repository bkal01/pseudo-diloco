import copy
import torch

from transformers import get_cosine_schedule_with_warmup

from src.pseudo_diloco.config import *


class PseudoDiloco:
    def __init__(
        self,
        model: torch.nn.Module,
        M: int,
        config: Config,
    ):
        self.config = config
        self.base_model = model.cpu()
        self.M = M
        self.replicas = []
        self.inner_optimizers = []
        self.schedulers = []
        self.initial_base_params = {name: param.clone().detach() for name, param in self.base_model.named_parameters()}

        self.active_replica = 0

        for _ in range(M):
            replica_copy = copy.deepcopy(self.base_model).cpu()
            self.replicas.append(replica_copy)

            inner_opt = torch.optim.AdamW(
                params=replica_copy.parameters(),
                lr=self.config.inner_optimizer_config.lr,
                weight_decay=self.config.inner_optimizer_config.weight_decay,
                betas=(self.config.inner_optimizer_config.b1, self.config.inner_optimizer_config.b2),
            )
            self.inner_optimizers.append(inner_opt)
            self.schedulers.append(
                get_cosine_schedule_with_warmup(
                    optimizer=inner_opt,
                    num_warmup_steps=self.config.scheduler_config.num_warmup_steps,
                    num_training_steps=self.config.scheduler_config.num_training_steps,
                )
            )

        self.outer_optimizer = torch.optim.SGD(
            params=self.base_model.parameters(),
            lr=self.config.outer_optimizer_config.lr,
            momentum=self.config.outer_optimizer_config.momentum,
            nesterov=self.config.outer_optimizer_config.nesterov,
        )
        self.sync_replicas()


    def get_active_model(self):
        return self.replicas[self.active_replica]

    def get_active_inner_optimizer(self):
        return self.inner_optimizers[self.active_replica]

    def get_active_scheduler(self):
        return self.schedulers[self.active_replica]

    def next_replica(self):
        pass

    @torch.no_grad()
    def sync_replicas(self):
        pass

    def outer_step(self):
        self.outer_optimizer.zero_grad()
        
        for name, param in self.base_model.named_parameters():
            diff_sum = torch.zeros_like(param)
            for replica in self.replicas:
                replica_param = dict(replica.named_parameters())[name]
                diff_sum += param - replica_param
            
            avg_diff = diff_sum / self.M
            param.grad = avg_diff
        
        self.outer_optimizer.step()
        self.outer_optimizer.zero_grad()
    
