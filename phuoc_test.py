
from main_src.utils import read_yaml
from main_src.models import build_model
opt = read_yaml(r"D:\phuoc_sign\main_src\options\train\train_s3d_model.yaml")
model = build_model(opt)