import torch

from data.loader import load_dataset
from esn.reservoir import Reservoir

def main():
    tensor = load_dataset('mc100')

    reservoir = Reservoir(
        input_size=202,
        hidden_size=100,
        num_layers=3,
        activation='tanh',
        leakage=1,
        input_scaling=0.2,
        rho=0.99,
        bias=True,
        kernel_initializer='uniform',
        recurrent_initializer='uniform',
        mode='intrinsic_plasticity',
        mu=0,
        sigma=0.25
    )
    reservoir.zero_grad()
    print(reservoir(tensor))
    

if __name__ == '__main__':
    main()