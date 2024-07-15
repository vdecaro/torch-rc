import torch


def seq_collate_fn(scenario: str = "stationary"):

    if scenario == "stationary":

        def _collate_fn(batch):
            x, y = [], []
            for x_i, y_i in batch:
                x.append(x_i)
                y.append(y_i)

            return torch.stack(x, dim=1), torch.stack(y, dim=1)

    elif scenario == "continual":

        def _collate_fn(batch):
            x, y = [], []
            for x_i, y_i, _ in batch:
                x.append(x_i)
                y.append(y_i)

            return torch.stack(x, dim=1), torch.stack(y, dim=1)

    else:
        raise ValueError(f"Unknown scenario for collate_fn: {scenario[:10]}")

    return _collate_fn
