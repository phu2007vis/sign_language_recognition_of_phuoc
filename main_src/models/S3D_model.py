from main_src.utils.registry import MODEL_REGISTRY
from main_src.models.base_model import BaseModel
from main_src.architecture_network import build_network
import torch.nn as nn
from main_src.utils import logger
@MODEL_REGISTRY.register()
class S3D(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        print("creating model")
        self.net = build_network(opt['net']).to(opt['device'])
        load_path = self.opt['path'].get('pretrain_model', None)
        if load_path is not None:
            self.load_network(self.net,load_path)
        self.optimizers = [self.get_optimizer(self.net.parameters(),opt['train']['lr'],weight_decay = opt['train']['weight_decay'])]
        self.setup_schedulers()
        self.loss_fn = nn.CrossEntropyLoss()
    def feed_data(self,data):
        self.inputs,self.labels = data
    def optimize_parameters(self):
        self.optimizers[0].zero_grad()
        self.inputs = self.inputs.to(self.opt['device'])
        self.labels = self.labels.to(self.opt['device'])
        preds = self.net(self.inputs)
        self.loss = self.loss_fn(preds, self.labels)
        self.loss.backward()
        self.optimizers[0].step()
        del self.inputs
        del self.labels
    def get_current_lr(self):
        self.scheduler[0].get_last_lr()[0]
    


        
        
            
        
