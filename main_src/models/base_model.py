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

    def get_current_visuals(self):
        pass

    def save(self, epoch):
        """Save networks and training state."""
        pass

    def validation(self, dataloader:torch.utils.data.DataLoader):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
        """
        pass
    def save_network(self,net,save_path):
        pass
    def load_network(self, net, load_path):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
        """
        pass
    
                
    def get_optimizer(self, params, lr, **kwargs):
        '''
        Adam optimizer
        '''
        return torch.optim.Adam(params, lr, **kwargs)

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')
        

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

 




