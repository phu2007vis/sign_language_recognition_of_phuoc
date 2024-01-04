from main_src.utils.registry import MODEL_REGISTRY
from main_src.models.base_model import BaseModel
from main_src.architecture_network import build_network
 
@MODEL_REGISTRY.register()
class S3D(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        load_path = self.opt['path'].get('pretrain_model', None)
        self.net = build_network(opt['net'])
        if load_path is not None:
            self.load_network()
        
            
        
