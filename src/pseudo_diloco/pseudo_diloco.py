import copy
import torch

from src.pseudo_diloco.config import *
from src.pseudo_diloco.all_reduce import all_reduce

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=inner_opt,
                    T_max=self.config.scheduler_config.T_max,
                )
            )

        self.outer_optimizer = torch.optim.SGD(
            params=self.base_model.parameters(),
            lr=self.config.outer_optimizer_config.lr,
            momentum=self.config.outer_optimizer_config.momentum,
            nesterov=self.config.outer_optimizer_config.nesterov,
        )
        self.sync_replicas()
        self.replicas[self.active_replica].to(device)


    def get_active_model(self):
        return self.replicas[self.active_replica]

    def get_active_inner_optimizer(self):
        return self.inner_optimizers[self.active_replica]

    def get_active_scheduler(self):
        return self.schedulers[self.active_replica]

    def iterate_replica(self):
        self.active_replica = (self.active_replica + 1) % self.M

    @torch.no_grad()
    def sync_replicas(self):
        for replica in self.replicas:
            replica.load_state_dict(self.base_model.state_dict())

    def outer_step(self):
        self.outer_optimizer.zero_grad()
        
        all_reduce(
            base_model=self.base_model,
            replicas=self.replicas,
        )
        
        self.outer_optimizer.step()
        self.outer_optimizer.zero_grad()
    
