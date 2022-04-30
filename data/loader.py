import os
import csv
import torch

def load_dataset(name: str):
    module_path = os.path.dirname(__file__)
    if name == 'mc100':
        reader = csv.reader(open(os.path.join(module_path, 'mc100.csv')), delimiter=',')
        output = [
            list(map(lambda x: float(x), row)) for row in reader if row[0][0] != '#'
        ]
    
    return torch.tensor(output)


