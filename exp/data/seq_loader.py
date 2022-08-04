import torch


def seq_collate_fn(batch):
     x, y = [], []
     for x_i, y_i in batch:
          x.append(x_i)
          y.append(y_i)
     
     return torch.stack(x, dim=1), torch.stack(y, dim=1)