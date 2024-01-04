import torch.nn as nn
from torchvision.models.video import s3d
from main_src.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class S3D_arch(nn.Module):
    def __init__(self, num_classes: int,pretrained:bool = True, dropout: float = 0.2, **kwargs: any, *args: any):
        super(S3D_arch, self).__init__()
        self.main_net = s3d(num_classes=num_classes,pretrained, dropout=dropout, **kwargs, *args)

    def forward(self, x:torch.tensor):
        assert(len(x.shape)==5)
        return self.main_net(x)