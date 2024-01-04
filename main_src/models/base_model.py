import os
import time
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel

from main_src.models import lr_scheduler as lr_scheduler
from main_src.models import logger

class BaseModel():
    """Base model."""

    def __init__(self, opt):
        self.opt = opt
        self.device = opt['device']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def validation(self, dataloader:torch.utils.data.DataLoader):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
        """
        pass
    def save_network(self, net, save_path):
        """
        Save the PyTorch model to a file.

        Args:
            net (nn.Module): The PyTorch model to be saved.
            save_path (str): The file path to save the model.
        """
        torch.save(net.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_network(self, net, load_path):
        """
        Load a PyTorch model from a file.

        Args:
            net (nn.Module): The PyTorch model to be loaded into.
            load_path (str): The file path to load the model from.
        """
        try:
            net.load_state_dict(torch.load(load_path))
            print(f"Model loaded from {load_path}")
        except FileNotFoundError:
            print(f"Model file not found at {load_path}")
        except Exception as e:
            print(f"Error loading model from {load_path}: {e}")
                
    def get_optimizer(self, params, lr, **kwargs):
        '''
        Adam optimizer
        '''
        return torch.optim.Adam(params, lr, **kwargs)


    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        
        scheduler_class = getattr(lr_scheduler, scheduler_type, None)
        
        if scheduler_class is not None and callable(scheduler_class):
            for optimizer in self.optimizers:
                self.schedulers.append(scheduler_class(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

 




