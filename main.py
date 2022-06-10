import os
from tqdm import tqdm
import yaml
import datetime
import argparse
from shutil import copy2
import numpy as np


import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.model import MobileNetV3SkipAdd

from data import Places_Dataset, NyuDepthV2_Dataset
from loss import DepthLoss

from utils import MetricAggregator, disparity_to_depth, get_images, write_depth


def validate(val_loader, model, device):
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # validate
    metric_aggregator = MetricAggregator()

    t = tqdm(val_loader)

    with torch.no_grad():
        for i, batch in enumerate(t):

            # to device
            for k, v in batch.items():
                batch[k] = v.to(device)

            # run model
            prediction = model.forward(batch["image"])
            # resize prediction to match target

            
            prediction = (
                    F.interpolate(
                    prediction,
                    size=batch["mask"].shape[1:],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze(1)
            )
            
            prediction = disparity_to_depth(prediction, batch["depth"], batch["mask"])

            depth = np.squeeze(prediction.cpu().numpy())
            write_depth("results/"+str(i), depth)
            metric_aggregator.evaluate(prediction, batch["depth"], batch["mask"])
    
    metric_dict = metric_aggregator.average()

    msg_1 = "Validation on test samples:"
    msg_2 = f"RMSE: {metric_dict['rmse']:.3f} RMSE_log: {metric_dict['rmse_log']:.3f} REL: {metric_dict['absrel']:.3f} "
    msg_3 = f"Delta1: {metric_dict['delta1']:.3f} Delta2: {metric_dict['delta2']:.3f} Delta3: {metric_dict['delta3']:.3f}" 
    print(msg_1)
    print(msg_2)
    print(msg_3)
    
    return metric_dict['rmse']

def train_epoch(train_loader, model, optimizer, epoch, device, config, log_file):
    t = tqdm(train_loader)
    running_loss = 0.0
    running_mse_loss = 0.0
    running_grad_loss = 0.0
    
    for i, data in enumerate(t):
        rgb, depth_gt = data
        rgb = rgb.to(device)
        depth_gt = depth_gt.to(device)
        
        # forward pass

        depth_pred = model(rgb)
        depth_gt = depth_gt.unsqueeze(1)

        # loss part
        optimizer.zero_grad()
        loss = criterion(depth_pred, depth_gt)

        loss.backward()

        running_loss += loss.item()
        running_mse_loss += criterion.last_mse
        running_grad_loss += criterion.last_grad

        optimizer.step()

        # record loss

        if (i + 1) % config['log_interval'] == 0:
            avg_loss_value = running_loss / config['log_interval']
            avg_mse_value = running_mse_loss / config['log_interval']
            avg_grad_value = running_grad_loss / config['log_interval']

            msg = f"[{epoch+1}, {i+1}] Loss: {avg_loss_value:.5f}, MAE: {avg_mse_value:.5f}, GRAD: {avg_grad_value:.5f}"
            print(msg)

            with open(log_file, 'a') as f:
                print(msg, file=f)

            running_loss = 0.0
            running_mse_loss = 0.0
            running_grad_loss = 0.0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader = yaml.FullLoader)

    # random.seed(config["random_seed"])
    # np.random.seed(config["random_seed"])
    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed(config["random_seed"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("Device: {}".format(device))

    batch_size = config['batch_size']

    test_flag = config['test_flag']

    workers = config['workers']

    nyu_data = config['nyu_data_path']

    nyu_split = config['nyu_split_path']

    pretrained_enc = config['pretrained_enc']

    model = MobileNetV3SkipAdd(pretrained_enc)
    model = model.to(device)

    if test_flag:        
        print("Initialize model") 

        model.load_state_dict(torch.load(config['best_model']))
        
        model.eval()
        
        test_dataset = NyuDepthV2_Dataset(nyu_data, nyu_split, config)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)

        validate(test_loader, model, device)
   
    else:
        
        train_filenames = get_images(config['train_filepath'])
        print(f"Dataset size: {len(train_filenames)}")

        train_dataset = Places_Dataset(train_filenames)
        val_dataset = NyuDepthV2_Dataset(nyu_data, nyu_split, config)
          

        print("Initialized datasets")    
        
        # 2. initialize dataloaders
        train_loader = DataLoader(train_dataset, batch_size, True, num_workers=workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)

        parameters = list(model.parameters())


        # 4. initialize criterions 
        criterion_name = config['criterion']

        if criterion_name == 'l1':
            criterion = torch.nn.L1Loss().to(device)
        elif criterion_name == 'depth':
            criterion = DepthLoss().to(device)

        print("Initialized criterion")

        # 5. initialize optimizers
        opt_name = config['optimizer']
        opt_params = config['opt_params']
        # opt_params example {'lr':0.01, 'weight_decay':1e-4, 'momentum': 0.9}
        
        if opt_name == 'adam':
            optimizer = torch.optim.Adam(parameters, **opt_params)
        elif opt_name == 'sgd':
            optimizer = torch.optim.SGD(parameters, **opt_params)

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config['lr_decay_rate'])
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.1,)

        print("Initialized optimizer")

        resume_epoch = 0
        if(config['resume']):
            checkpoint = torch.load(config['resume_path'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            resume_epoch = checkpoint['epoch']
            print("Checkpoint loaded")
        
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        
        # initialize folders to store models, etc    

        current_date = datetime.datetime.now().strftime('%d%m%y')
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_path = f"{config['checkpoint_path']}/{current_date}/{config['run_name']}_{current_time}" 
        
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        print("Initialized store folders")

        logfile_path = os.path.join(model_path, 'log.txt')

        with open(logfile_path, 'w') as f:
            print("", file=f)

        copy2(args.config, model_path)

        num_epochs = config['n_epochs']

        print("Validate before training:")

        model.eval()
        validate(val_loader, model, device)
        print("Start Training:")
        inv_best_acc = 0.0
        for epoch in range(resume_epoch, num_epochs):
            print("Epoch: {}".format(epoch+1))
            model.train()

            train_epoch(train_loader, model, optimizer, epoch, device, config, logfile_path)
            
            print("Validating:")
            model.eval()

            epoch_acc = validate(val_loader, model, device)
            inv_epoch_acc = 1/epoch_acc

            if inv_epoch_acc > inv_best_acc:
                inv_best_acc = inv_epoch_acc
                torch.save(model.state_dict(), f"{model_path}/model_{1/inv_best_acc}.pth")
            lr_scheduler.step()

            torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        }, f"{model_path}/checkpoint.pth")
        

        print("Successfully saved model")
        print("------------------------")
