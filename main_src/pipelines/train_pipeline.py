import torch
from os import path as osp
from main_src.utils import logger
from main_src.data_preparation.DataLoader import buid_dataloader
from main_src.models import build_model
from main_src.utils import read_yaml
import tqdm
import os

def create_train_val_dataloader(opt):
    # create train and val dataloaders
    train_loader, val_loader = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':       
            train_loader = buid_dataloader(**dataset_opt)     
        elif phase == 'val':
            val_loader = buid_dataloader(**dataset_opt)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
    return train_loader, val_loader

def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt =  read_yaml(root_path)
    opt['root_path'] = root_path
    torch.backends.cudnn.benchmark = True
    # create train and validation dataloaders
    train_loader, val_loaders = create_train_val_dataloader(opt)
    if not train_loader:
        print("Please provide train data")
        exit()
    model = build_model(opt)
    
    print(model.net)
    logger.info('build model succesfull')
    start_epoch = 0
    current_iter = 0
    total_iters = opt['train']['total_iter']
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    
    while True:
        for inputs,labels in  train_loader:
            current_iter += 1
            if current_iter > total_iters:
                break
            model.net.train()
            # training
            train_data = (inputs,labels)
            model.feed_data(train_data)
            model.optimize_parameters()
            model.update_learning_rate()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': start_epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                path = os.path.join(model.opt['logger']['save_folder'],"iter_"+str(current_iter)+".pth")
                logger.info('Saving models and training states.')
                model.save_network(model.net,path)

            # # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                pass
    path = os.path.join(model.opt['logger']['save_folder'],"iter_"+str(-1)+".pth")
    logger.info('Saving models and training states.')
    model.save_network(model.net,path)
    if opt.get('val') is not None:
       pass


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(r"D:\phuoc_sign\main_src\options\train\train_s3d_model.yaml")
