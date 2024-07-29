import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchrc.data.utils.seq_loader import seq_collate_fn
from torchrc.models.rnn2.rnn2 import RNN2Layer
from torchrc.models.initializers import sparse, orthogonal
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from torchrc.data.datasets.seq_mnist import SequentialMNIST
from torch.utils.data import Subset
from tqdm import tqdm

# RNN of RNNs params
BLOCKS = [32 for _ in range(16)]
HSIZE = sum(BLOCKS)
RNN_DENSITY = 0.03
COUPLINGS = [(i, j) for i in range(16) for j in range(16) if i < j]
random.shuffle(COUPLINGS)
COUPLINGS = COUPLINGS[:20]
EUL_STEP = 0.03

BLOCK_INIT_FN = lambda x: sparse(x, RNN_DENSITY)
COUPLE_INIT_FN = lambda x: orthogonal(x, 0.9)


# Training params
EPOCHS = 200
LR = 1e-3
LAST_EPOCH = 100
LR_SCALAR = 0.1
WEIGHT_DECAY = 1e-5
TRAIN_BS = 64
TEST_BS = 1024

output_root = "./results"


def load_mnist():
    dataset = SequentialMNIST("./data", train=True, download=True)

    offset = 2000
    rng = np.random.RandomState(1234)
    perm_idx = rng.permutation(len(dataset))
    train_idx, val_idx = perm_idx[offset:], perm_idx[:offset]
    train_data, val_data = Subset(dataset, train_idx), Subset(dataset, val_idx)
    test_data = SequentialMNIST("./data", train=False, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=TRAIN_BS, shuffle=True, collate_fn=seq_collate_fn()
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=TEST_BS, shuffle=False, collate_fn=seq_collate_fn()
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=TEST_BS, shuffle=False, collate_fn=seq_collate_fn()
    )
    return train_loader, val_loader, test_loader


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rnn = RNN2Layer(
        input_size=1,
        out_size=10,
        block_sizes=BLOCKS,
        coupling_indices=COUPLINGS,
        block_init_fn=BLOCK_INIT_FN,
        coupling_block_init_fn=COUPLE_INIT_FN,
        eul_step=EUL_STEP,
        activation="relu",
    )
    rnn.to(device)
    optim_params = list(rnn.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(optim_params, lr=LR, weight_decay=WEIGHT_DECAY)
    lr = ExponentialLR(optimizer, gamma=LR_SCALAR)
    test_accs = []
    train_losses = []
    epochs_list = []  # just grabbing numbers for the sake of dataframe
    train_loader, val_loader, _ = load_mnist()
    # Train for some epochs
    for epoch in tqdm(range(EPOCHS), total=EPOCHS):
        rnn.train()
        loss_epoch = []
        for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            # had to add these lines for colab GPU
            x = x.to(device)
            y = y.to(device)

            pred, _ = rnn(x)

            loss = criterion(pred[-1], y)
            loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
        lr.step()
        # track loss over time, and have it reflect the mean loss over the batches so it is more reflective of training trends than last batch loss
        mean_loss = np.mean(loss_epoch)
        train_losses.append(mean_loss)

        # calling the network with no training epoch 0, want each epoch number to reflect how many have been run so far, so add 1 here
        print("Epoch {}, mean batch loss {}".format(epoch + 1, mean_loss))

        # testing is fast so no problem doing it every epoch
        rnn.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in val_loader:
                x = x.cuda()  # adding here too
                y = y.cuda()
                pred, _ = rnn(x)

                # the class with the highest energy is what we choose as prediction
                predicted = torch.argmax(pred[-1], -1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            print(
                "Accuracy of the network on the 10000 test images: %d %%"
                % (100 * correct / total)
            )
            test_accs.append(
                (100.0 * float(correct) / float(total))
            )  # for the actual list make sure we are getting exact test accuracy, so convert to floats!
        epochs_list.append(epoch + 1)  # end of epoch so label +1

        # save model for every epoch as storage space required is quite small, can have training disruptions with colab
        model_path = os.path.join(
            output_root,
            "rnn2-mnist-epoch" + str(epoch + 1) + ".pt",
        )  # end of epoch so label +1
        rnn.save(model_path)  # would use torch.jit.load to reload in the future

        # save stats so far, will overwrite every time as rows are added to dataframe
        stats_path = os.path.join(output_root, "rnn2-mnist-stats.csv")
        cur_stats = pd.DataFrame()
        cur_stats["epoch"] = epochs_list
        cur_stats["loss"] = train_losses
        cur_stats["test-acc"] = test_accs
        cur_stats.to_csv(stats_path, index=False)

    return rnn, optimizer, test_accs, train_losses


if __name__ == "__main__":
    train()
