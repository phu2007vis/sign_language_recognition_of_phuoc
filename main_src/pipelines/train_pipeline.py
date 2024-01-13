import torch
from os import path as osp
from main_src.data_preparation.DataLoader import buid_dataloader
from main_src.models import build_model
from main_src.utils import read_yaml
import tqdm
import os
from main_src.utils.validation import Validation
from main_src.utils import *
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
    opt = read_yaml(root_path)
    torch.backends.cudnn.benchmark = True
    print("creating dataloader")
    logger.info("creating dataloader")
    train_loader, val_loader = create_train_val_dataloader(opt)  # Modify this line to suit your data loading structure

    if not train_loader:
        logger.info("Please provide train data")
        exit()

    model = build_model(opt)
    logger.info('Build model successful')
    print("Build model successful")

    start_epoch = 0
    current_iter = 0
    total_iters = opt['train']['total_iter']
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    print("Start training ")

    # Validation object
    validation_handler = Validation( opt=opt)

    while True:
        start_epoch+=1
        for inputs, labels in train_loader:
            current_iter += 1
            if current_iter > total_iters:
                break

            model.net.train()
            # training
            train_data = (inputs, labels)
            model.feed_data(train_data)
            model.optimize_parameters()
            model.update_learning_rate()

            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                # log_vars.update(model.get_current_log())
                # Print your log_vars here
                print(f"Iter: {current_iter} loss: {model.loss}")
                logger.info(f"Iter: {current_iter} loss: {model.loss}")

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                path = os.path.join(model.opt['logger']['save_folder'], f"iter_{current_iter}.pth")
                logger.info('Saving models and training states.')
                model.save_network(model.net, path)

            del train_data
            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                precision, recall, f1, average_loss,accuracy = validation_handler.validate(model.net, val_loader, model.loss_fn)
                logger.info("validate")
                logger.info(f"average_loss: {average_loss} , accuracy: {accuracy} in validate set")
                print("validate")
                print("average_loss: {average_loss}")
                # Log or save the validation metrics as needed

    # Save models and training states at the end of training
    path = os.path.join(model.opt['logger']['save_folder'], f"iter_{-1}.pth")
    logger.info('Saving models and training states.')
    model.save_network(model.net, path)

    if opt.get('val') is not None:
        precision, recall, f1, average_loss = validation_handler.validate(model, val_loader, criterion)
        # Log or save the final validation metrics as needed


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(r"/work/21013187/phuoc_sign/sign_language_recognition_of_phuoc/main_src/options/train/train_s3d_model.yaml")
