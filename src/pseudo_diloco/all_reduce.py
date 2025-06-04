import torch

def all_reduce(
        base_model: torch.nn.Module,
        replicas: list[torch.nn.Module],
    ):
    for name, param in base_model.named_parameters():
        diff_sum = torch.zeros_like(param)
        for replica in replicas:
            replica_param = dict(replica.named_parameters())[name]
            diff_sum += param - replica_param
        
        avg_diff = diff_sum / len(replicas)
        param.grad = avg_diff