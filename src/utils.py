import torch
import random
import numpy as np
import logging
import os

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str, epoch: int):
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(state, latest_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        torch.save(state, best_path)

def load_checkpoint(checkpoint_path: str, model, optimizer=None, scheduler=None):
    if not os.path.isfile(checkpoint_path):
        return 0, 0.0
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    return state.get('epoch', 0), state.get('best_metric', 0.0)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {'total': total, 'trainable': trainable, 'frozen': frozen}

class EarlyStopping:
    def __init__(self, patience, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_metric = -float('inf') if mode == 'max' else float('inf')
        self.early_stop = False

    def __call__(self, metric):
        if self.mode == 'max':
            if metric >= self.best_metric + self.min_delta:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1
        else:
            if metric <= self.best_metric - self.min_delta:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop

    def reset(self):
        self.counter = 0
        self.best_metric = -float('inf') if self.mode == 'max' else float('inf')
        self.early_stop = False

class MetricTracker:
    def __init__(self):
        self.metrics = {}
        
    def reset(self):
        self.metrics = {}
        
    def update(self, metrics_dict, n=1):
        for k, v in metrics_dict.items():
            if k not in self.metrics:
                self.metrics[k] = {'sum': 0.0, 'count': 0}
            self.metrics[k]['sum'] += v * n
            self.metrics[k]['count'] += n
            
    def compute(self):
        return {k: v['sum']/v['count'] for k, v in self.metrics.items()}
