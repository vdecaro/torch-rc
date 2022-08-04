import argparse
import os

import ray
from exp.phase import model_selection, retraining, test

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--mode', '-m', type=str)
parser.add_argument('--percentage', '-p', type=int, default=100)
parser.add_argument('--gpu_trial', '-g', type=int, default=1)



def main():
    args = parser.parse_args()
    dataset, perc, gt, mode = args.dataset, args.percentage, args.gpu_trial, args.mode

    exp_dir = f"experiments/{config['DATASET']}_{perc}_{mode}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    ray.init()
    
    model_selection.run(dataset, perc, mode, gt)
    retraining.run(dataset, perc, mode, gt, exp_dir)

if __name__ == '__main__':
    main()