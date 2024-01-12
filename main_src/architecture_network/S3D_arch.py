import torch.nn as nn
from torchvision.models.video import s3d
from main_src.utils.registry import ARCH_REGISTRY
import torch
@ARCH_REGISTRY.register()
class S3D_arch(nn.Module):
    def __init__(self, num_classes: int ,dropout: float = 0.2,pretrained:bool = True, **kwargs: any):
        super(S3D_arch, self).__init__()
        self.main_net = s3d(num_classes=400, dropout=dropout)
        self.main_net.load_state_dict(torch.load("/work/21013187/phuoc_sign/sign_language_recognition_of_phuoc/main_src/models/s3d-d76dad2f.pth"))
        self.main_net.classifier[1] = nn.Conv3d(1024, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    def forward(self, x):
        assert(len(x.shape)==5)
        return self.main_net(x)
    
