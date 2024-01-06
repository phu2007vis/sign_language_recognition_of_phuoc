import torch
from copy import deepcopy
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
            net.load_state_dict(torch.load(load_path, map_location=torch.device(self.opt['device'])))
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
    def update_learning_rate(self):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        for scheduler in self.schedulers:
            scheduler.step()
      

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


 




