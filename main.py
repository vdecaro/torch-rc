import argparse
import os
os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '10'
import ray
from exp.phase import model_selection, retraining, test

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--mode', '-m', type=str)
parser.add_argument('--percentage', '-p', type=int, default=100)
parser.add_argument('--gpu_trial', '-g', type=float, default=1)


def main():
    args = parser.parse_args()
    dataset, perc, gt, mode = args.dataset, args.percentage, args.gpu_trial, args.mode

    ray.init(local_mode=True)    
    model_selection.run(dataset, perc, mode, gt)
    retraining.run(dataset, perc, mode, gt)
    test.run(dataset, perc, mode, gt)

if __name__ == '__main__':
    main()