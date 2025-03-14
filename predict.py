import torch
#import torchvision
from torch import nn
from torch.utils.data import DataLoader,Dataset,random_split
import numpy as np
import argparse
import os
import xarray as xr
from datetime import datetime,timedelta
import pandas as pd
from functools import partial
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
from swin_unet import SwinTransformerSys
from GLFNet import GLFNet
from swinlstm import SwinLSTM

def model_forward_multi_layer(model, inputs, targets_len, num_layers):
    states_down = [None] * len(num_layers)
    states_up = [None] * len(num_layers)

    outputs = []

    inputs_len = inputs.shape[1]

    last_input = inputs[:, -1]

    for i in range(inputs_len - 1):
        output, states_down, states_up = model(inputs[:, i], states_down, states_up)
        #outputs.append(output)

    for i in range(targets_len):
        output, states_down, states_up = model(last_input, states_down, states_up)
        outputs.append(output)
        last_input = output

    return outputs

def are_all_paths_exist(file_paths):
    for path in file_paths:
        if not os.path.exists(path):
            print(path, ' not exists')
            return False
    return True

def create_filepath(year, month, day):
   
    year = str(year).zfill(4)
    month = str(month).zfill(2)
    day = str(day).zfill(2)
    
    file_root = '/work1/licom/ztao/dl_project/diff_way_sst_forecast/data/OISST'
    file_name = f'{year}-{month}-{day}.nc'
    all_file_path = os.path.join(file_root, file_name)

    return [all_file_path]

def read_current_day_data(year, month, day):
    
    all_file_path = create_filepath(year, month, day)
    all_file_path = all_file_path[0]
    sst = xr.open_dataset(all_file_path).sst
    
    return np.array(sst) 

def sample_create(start_time, input_step, output_step):

    # start_time is previous day
    previous_day_path = []
    for i in range(input_step-1,-1,-1):
        current_date = start_time - timedelta(days=i)
        year = current_date.year
        month = current_date.month
        day   = current_date.day
        previous_day_path = previous_day_path + create_filepath(year, month, day)

    future_day_path = []
    for i in range(1, output_step+1, 1):
        current_date = start_time + timedelta(days=i)
        year  = current_date.year
        month = current_date.month
        day   = current_date.day
        future_day_path = future_day_path + create_filepath(year, month, day)

    if are_all_paths_exist(previous_day_path) and are_all_paths_exist(future_day_path):
        sample_x          = []
        sample_y          = []
        for i in range(input_step-1,-1,-1):
            current_date = start_time - timedelta(days=i)
            year = current_date.year
            month = current_date.month
            day   = current_date.day
            sst = read_current_day_data(year, month, day)
            sample_x.append(sst)

        for i in range(1, output_step+1, 1):
            current_date = start_time + timedelta(days=i)
            year = current_date.year
            month = current_date.month
            day   = current_date.day
            sst = read_current_day_data(year, month, day)
            sample_y.append(sst)

        return np.array(sample_x)[np.newaxis, :, np.newaxis, :, :],\
               np.array(sample_y)[np.newaxis, :, np.newaxis, :, :], 1 
    else:
        print(start_time, ' data have problems')
        return None, None, 0

def fill_nan(x):
    x[np.isinf(x)] = 0 
    x[np.isnan(x)] = 0 
    return np.array(x)

def predict(x_input, model, x_mean, x_std, device = 'cpu', recurrent_step=1):

    x_input = torch.from_numpy(x_input.astype(np.float32))
    x_mean = torch.from_numpy(x_mean.astype(np.float32)).to(device)
    x_std = torch.from_numpy(x_std.astype(np.float32)).to(device)
    model.eval()
    model_pre_all = []
    for i in range(recurrent_step):
        with torch.no_grad():
            x_input = x_input.to(device)
            #pred = model(x_input)
            pred = torch.stack(model_forward_multi_layer(model, x_input, 10, [2,6]), axis=1)
            x_input = pred

        if len(pred.shape)!=5:
            pred = pred[:,:,np.newaxis,:,:]

        pred = pred.to(device)
        pred_reverse = pred*x_std + x_mean 
        model_pre_all.append(pred_reverse)
   
    return torch.cat(model_pre_all, axis=1) 


def ncfile_save(data, filepath, lon, lat, lev, lead_days):
    ds = xr.DataArray(np.array(data, dtype=np.dtype('f4')), coords=[lead_days, lev, lat, lon], dims = ['lead_days', 'lev', 'lat','lon'])
    ds_set = xr.Dataset({'sst':ds})
    ds_set.to_netcdf(filepath)


def case_forecast(model, x_mean, x_std, dim=5):

    if not os.path.exists('./forecast_file'): os.mkdir('./forecast_file')
    target_path = '/work1/licom/ztao/dl_project/diff_way_sst_forecast/data/OISST/1982-01-17.nc'
    target_tt = xr.open_dataset(target_path)['sst']
    lon = target_tt.lon
    lat = target_tt.lat
    lev = range(1)
    lead_days = range(1,11,1)
    start_time = datetime(2020,1,10)
    
    for i in range(3*365-20):
        sample_x, sample_y, flag = sample_create(start_time, input_step, output_step)
        x_10 = sample_x[:, 9:10, :, :, :]
        x_10 = torch.from_numpy(x_10).to(0)
        sample_x = (sample_x - x_mean)/x_std
        sample_x = fill_nan(sample_x)
        if dim ==5 or dim==3:
            pass
        elif dim ==4:
            sample_x = sample_x.squeeze(axis=2)
         
        model_pre = predict(sample_x, model, x_mean, x_std, device = 'cuda', recurrent_step=1)


        current_str = "{:%Y-%m-%d}".format(start_time+timedelta(days=1))
        obs_path = f'./forecast_file/' + 'obs_' + current_str + '.nc'
        ai_path = obs_path.replace('obs', 'ai')
        if dist.get_rank() == 0:
            ncfile_save(np.array(model_pre.squeeze(axis=0).cpu()), ai_path, lon, lat, lev, lead_days)
            ncfile_save(np.array(sample_y.squeeze(axis=0)), obs_path, lon, lat, lev, lead_days)
        if i%10==0:
            print(f'{obs_path} finished!')
            print('')
        start_time = start_time + timedelta(days=1)

class MaskedMSELoss(nn.MSELoss):
    def __init__(self, mask):
        super(MaskedMSELoss, self).__init__()
        self.mask = mask
    def forward(self, preds, labels):
        loss = (preds - labels) ** 2
        loss = loss * self.mask
        return torch.mean(loss)

def dist_init(host_addr, rank, local_rank, world_size, port=1234):
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    print("host_addr_full: ",host_addr_full)
    torch.distributed.init_process_group("nccl", init_method=host_addr_full, rank=rank, world_size=world_size)
    assert torch.distributed.is_initialized()

class ModelConfig:
    def __init__(self, model_name='predrnn', pretrained_model='', num_hidden='4,4', img_channel=1, img_width=(64),
                 filter_size=5, stride=1, patch_size=4, layer_norm=1, decouple_beta=0.1,
                 total_length=20, input_length=10):
        self.model_name = model_name
        self.pretrained_model = pretrained_model
        self.num_hidden = [int(x) for x in num_hidden.split(',')]  # 转换为整数列表
        self.filter_size = filter_size
        self.stride = stride
        self.patch_size = patch_size
        self.layer_norm = bool(layer_norm)  # 将 1 或 0 转换为布尔值
        self.decouple_beta = decouple_beta
        self.num_layers  = len(self.num_hidden)
        self.img_channel = img_channel
        self.img_width   = img_width
        self.total_length = total_length
        self.input_length = input_length

    def display(self):
        print(f"Model Name: {self.model_name}")
        print(f"Pretrained Model: {self.pretrained_model}")
        print(f"Number of Hidden Layers: {self.num_hidden}")
        print(f"Filter Size: {self.filter_size}")
        print(f"Stride: {self.stride}")
        print(f"Patch Size: {self.patch_size}")
        print(f"Layer Normalization: {self.layer_norm}")
        print(f"Decouple Beta: {self.decouple_beta}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--e', type=int, default=300)
    parser.add_argument('--op', type=str, default='./results/loss.npy')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--mp', type=str, default="model.pth")
    parser.add_argument("--master_ip", default=None, type=str)
    parser.add_argument("--world_size", default=None, type=int)
    parser.add_argument("--rank", default=None, type=int)
    parser.add_argument("--port", default=1234, type=int)
    args = parser.parse_args()
    dist_init(args.master_ip, args.rank, args.local_rank, args.world_size, args.port)
    obs_leads = 10
    device = args.local_rank
    netname = ['ConvLSTM','AFNONet','DNNNet','AttU_Net','R2AttU_Net','UNet','ConvLSTM_UNet','CNN']
    model_id = 0
    input_step=10
    output_step=10

    model = SwinLSTM(img_size=(720,1440), patch_size=(4,4),
                        in_chans=1, embed_dim=12,
                        depths_downsample=[2,2], depths_upsample=[2,2],
                        num_heads=[4,8], window_size=5).to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True, broadcast_buffers=False,)

    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    modelpath = f'./results/{netname[model_id]}_model_epoch52.pth'
    model.load_state_dict(torch.load(modelpath))
    print('Model has load!')
    x_mean = np.load('../data/npy/sst_mean.npy')[:,-input_step:,:,:,:]
    x_std = np.load('../data/npy/sst_std.npy')[:,-input_step:,:,:,:]
    case_forecast(model, x_mean, x_std, dim=5)
