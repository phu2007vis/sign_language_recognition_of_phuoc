import numpy as np
import torch
from pathlib2 import Path
from torch.utils.data import Dataset,DataLoader
from main_src.utils.temporal_transforms import *
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


class SpacialTransform(Dataset):
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size
        #augument image and normalize it
        self.transform = transforms.Compose([
            transforms.Resize(output_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def transform_fn(self, image_nps):
        image_PILs = []
        for image_np in image_nps:
                image_PIL = Image.fromarray(image_np.astype('uint8'))
                image_PIL = self.transform(image_PIL)
                image_PILs.append(image_PIL)
        return torch.stack(image_PILs)


class VideoFloderDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, sample_type="num", fps=5, out_frame_num=32):
        """
        Args:
            root_dir (string): Directory with all the video.
        """
        self.root_dir = Path(root_dir)
        self.sub_dirs = [i for i in self.root_dir.iterdir() if i.is_dir() and not i.stem.startswith('.')]
        self.class_names = [i.stem for i in self.sub_dirs]
        self.labels = []
        self.datas = []
        self.spacial_transform = SpacialTransform()
        self.temporal_transform = TemporalRandomCrop(out_frame_num)
        self.sample_type = sample_type
        for label,sub_dir in enumerate(self.sub_dirs):
            # item_dirs = [i for i in sub_dir.iterdir() if i.is_dir() and not i.stem.startswith('.')]
            # for item_dir in item_dirs:
                contents = [i for i in sub_dir.iterdir() if i.is_file() and not i.stem.startswith('.')]
                if contents:
                    
                    try:
                        if sample_type == "num":
                            temp_rgb = [i for i in contents ]
                            self.labels.extend([label]*len(contents))
                        
                        elif sample_type == "fps":
                            temp_rgb = [i for i in contents if i.stem.startswith('rgb') and "FPS" in i.stem
                                        and int(i.stem.split('_')[-1]) == fps][0]
                        else:
                            raise ValueError("sample_type should be 'num' or 'fps'")
                    except IndexError:
                        raise IndexError("Please make sure you specified input sample num or fps right.")
                    self.datas.extend(temp_rgb)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        rgb_data = np.float32(np.load(self.datas[idx]))
        rgb_data = self.temporal_transform(rgb_data)
        rgb_data = self.spacial_transform.transform_fn(rgb_data)
        rgb_data = rgb_data.permute(1,0,2,3).unsqueeze(0)
        return rgb_data,torch.nn.functional.one_hot(torch.tensor(self.labels[idx]), len(self.class_names)).unsqueeze(0)
    
def collate_fn(data):
    features, labels  = zip(*data)
    features = torch.cat(features,dim = 0)
    labels = torch.cat(labels,dim = 0)
    return features,labels

def get_dataloader(data_root,out_frame_num=32):
    dataset = VideoFloderDataset(data_root,out_frame_num=out_frame_num)
    return DataLoader(dataset, collate_fn=collate_fn, batch_size=4,shuffle = True)

if __name__ == "__main__":
    dataloader = get_dataloader(r"D:\phuoc_sign\dataset\raw_data")
    for image,label  in dataloader:
        print(image.shape)
        print(label.shape)
        pass
    