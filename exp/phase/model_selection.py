import os
from ray import tune
from typing import Literal
import exp.data.split_config as cfg

from ..utils.early_stopping import TrialNoImprovementStopper
from ..trainable import ESNTrainable


def run(dataset: str,
        perc: int,
        mode: Literal['vanilla', 'intrinsic_plasticity'],
        gt: float, 
        exp_dir: str):
    exp_dir = f"experiments/{dataset}_{perc}_{mode}"
    os.makedirs(exp_dir, exist_ok=True)
    config = get_config(dataset, perc)
    if mode == 'intrinsic_plasticity':
        stopper = TrialNoImprovementStopper(metric='eval_score', 
                                            mode='max', 
                                            patience_threshold=config['PATIENCE'])
    if mode == 'vanilla':
        stopper = lambda trial_id, result: True
    
    reporter = tune.CLIReporter(metric_columns={
                                    'training_iteration': '#Iter',
                                    'train_score': 'TR-Score',
                                    'eval_score': 'VL-Score', 
                                },
                                parameter_columns={'EPOCHS': '#E', 'SIGMA': 'SIGMA', 'HIDDEN_SIZE': '#H', 'SEQ_LENGTH': '#seq', 'LEAKAGE': 'alpha'},
                                infer_limit=3,
                                metric='eval_score',
                                mode='max')

    return tune.run(
        ESNTrainable,
        name=f"model_selection",
        stop=stopper,
        local_dir=exp_dir,
        config=config,
        num_samples=30,
        resources_per_trial={"CPU": 1, "GPU": gt},
        keep_checkpoints_num=1,
        checkpoint_score_attr='eval_score',
        checkpoint_freq=1,
        max_failures=5,
        progress_reporter=reporter,
        verbose=1,
        reuse_actors=True
    )


def get_config(name, perc, mode):
    d_users = cfg.USERS[name]
    if name == 'WESAD':
        config = {
            'DATASET': 'WESAD',
            'TRAIN_USERS': d_users['TRAIN'][perc],
            'VALIDATION_USERS': d_users['VALIDATION'],
            'SEQ_LENGTH': 700,
            'INPUT_SIZE': 8,
            'N_CLASSES': 4,

            'HIDDEN_SIZE': tune.choice([200, 300, 400]),
            'RHO': tune.uniform(0.3, 0.99),
            'LEAKAGE': tune.choice([0.1, 0.3, 0.5, 0.7, 0.8]),
            'INPUT_SCALING': tune.uniform(0.5, 1),
            'MU': 0,
            'SIGMA': tune.uniform(0.005, 0.15),
            'ETA': 1e-2,
            'L2': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1],
            'BATCH_SIZE': 100,
            'EPOCHS': tune.choice([1, 3, 5]),
            'PATIENCE': 5,
            'MODE': mode
        }
    if name == 'HHAR':
        config = {
            'DATASET': 'HHAR',
            'TRAIN_USERS': d_users['TRAIN'][perc],
            'VALIDATION_USERS': d_users['VALIDATION'],
            'SEQ_LENGTH': 400,
            'N_CLASSES': 6,
            'INPUT_SIZE': 6,

            'HIDDEN_SIZE': tune.choice([100, 200, 300, 400, 500]),
            'RHO': tune.uniform(0.3, 0.99),
            'LEAKAGE': tune.choice([0.1, 0.3, 0.5]),
            'INPUT_SCALING': tune.uniform(0.5, 1),
            'MU': 0,
            'SIGMA': tune.uniform(0.005, 0.15),
            'ETA': 1e-2,
            'L2': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1],
            'BATCH_SIZE': 50,
            'EPOCHS': tune.choice([1, 3, 5]),
            'PATIENCE': 5,
            'MODE': mode
        }
    
    return config