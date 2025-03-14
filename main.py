import torch
#import torchvision
from torch import nn
from torch.utils.data import DataLoader,Dataset,random_split
import numpy as np
import argparse
import os
#import xarray as xr
import math
from functools import partial
from model import UNet
from ed_convlstm import EF_Conv
import re
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
from swin_unet import SwinTransformerSys
from GLFNet import GLFNet
import json
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

class CustomDataset(Dataset,):
    def __init__(self, file_root_x, file_root_y):
        self.file_root_x = file_root_x
        self.file_root_y = file_root_y
        self.file_list = sorted(os.listdir(file_root_x))
        self.x_list = [i for i in self.file_list if i[0]=='x']
        self.y_list = [i.replace('x_input','y_output') for i in self.x_list]

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):

        x_path = os.path.join(self.file_root_x, self.x_list[idx])
        x_input = np.load(x_path)
        x_input = x_input.squeeze(axis=(0))
        y_path = os.path.join(self.file_root_y, self.y_list[idx])
        y_output = np.load(y_path)
        y_output = y_output.squeeze(axis=(0))

        return x_input, y_output

# Create dataset
def dataset(batch_size=16, obs_leads=10, model='ConvLSTM', dim = 5, input_step=1, output_step=1):

    train_dataset = CustomDataset(file_root_x = '../data/sst_x_train_data', file_root_y='../data/sst_y_train_data' )
    val_dataset = CustomDataset(file_root_x = '../data/sst_x_val_data', file_root_y='../data/sst_y_val_data' )
    test_dataset = CustomDataset(file_root_x = '../data/sst_x_test_data', file_root_y='../data/sst_y_test_data' )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=2)
    return train_loader, val_loader, test_loader
    
def train(dataloader, model, loss_fn, optimizer, device = 'cpu'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    data_loss = 0
    if dist.get_rank()==0:
        print(size)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        #pred = model(X)
        pred = torch.stack(model_forward_multi_layer(model, X, 10, [2,6]), axis=1)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        data_loss += loss.item()

        if batch % 10 == 0 and dist.get_rank()==0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    data_loss /= num_batches
    return data_loss

def index_cal(data_loader, model, loss_fn, device = 'cpu'):
    num_batches = len(data_loader)
    if dist.get_rank()==0:
        print("batch train: ", num_batches)
    model.eval()
    data_loss = 0
    data_ssim = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            #pred = model(X)
            pred = torch.stack(model_forward_multi_layer(model, X, 10, [2,6]), axis=1)
            data_loss += loss_fn(pred, y).item()
            #data_ssim += pytorch_ssim.ssim(pred,y)

    data_loss /= num_batches
    #data_ssim /= num_batches
    #return data_loss, data_ssim
    return data_loss, 1
    

def test(train_loss, train_loader, val_loader, test_loader, model, loss_fn, device = 'cpu'):

    #train_loss, train_ssim = index_cal(train_loader, model, loss_fn, device = device)
    train_ssim = 0
    val_loss, val_ssim = index_cal(val_loader, model, loss_fn, device = device)
    test_loss, test_ssim = index_cal(test_loader, model, loss_fn, device = device)

    if dist.get_rank()==0:
        print(f"Train Avg loss: {train_loss:>8f}, Train Avg ssim: {train_ssim:>4f} \n")
        print(f"Val Avg loss: {val_loss:>8f}, Val Avg ssim: {val_ssim:>4f} \n")
        print(f"Test Avg loss: {test_loss:>8f}, Test Avg ssim: {test_ssim:>4f} \n")

    return train_loss, val_loss, test_loss, train_ssim, val_ssim, test_ssim

class MaskedMSELoss(nn.MSELoss):
    def __init__(self, mask):
        super(MaskedMSELoss, self).__init__()
        self.mask = mask
    def forward(self, preds, labels):
        loss = (preds - labels) ** 2
        loss = loss * self.mask
        return torch.mean(loss)

class MaskedMSELoss2(nn.MSELoss):
    def __init__(self, mask):
        super(MaskedMSELoss2, self).__init__()
        self.mask = mask
    def forward(self, preds, labels):
        loss = torch.where(torch.gt(torch.abs(preds), torch.abs(labels)), (preds - labels) ** 2, 1.5*(preds - labels) ** 2)
        #loss = (preds - labels) ** 2
        loss = loss * self.mask
        return torch.mean(loss)

class EarthMSELoss(nn.MSELoss):
    def __init__(self, weights):
        super(EarthMSELoss, self).__init__()
        self.weights = weights

    def forward(self, preds, labels):
        loss = (preds - labels) ** 2
        loss = loss * self.weights
        return torch.sum(loss)

def dist_init(host_addr, rank, local_rank, world_size, port=1234):
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    print("host_addr_full: ",host_addr_full)
    torch.distributed.init_process_group("nccl", init_method=host_addr_full, rank=rank, world_size=world_size)
    assert torch.distributed.is_initialized()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--e', type=int, default=200)
    parser.add_argument('--op', type=str, default='./results/loss.npy')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--mp', type=str, default="model.pth")
    parser.add_argument("--master_ip", default=None, type=str)
    parser.add_argument("--world_size", default=None, type=int)
    parser.add_argument("--rank", default=None, type=int)
    parser.add_argument("--port", default=1234, type=int)
    #obs_leads = np.loadtxt('obs_leads.txt')
    args = parser.parse_args()
    dist_init(args.master_ip, args.rank, args.local_rank, args.world_size, args.port)
    #ds = xr.open_dataset('../data/area_info.nc')
    ds = np.load("../data/area_info.npz")
    weights = np.array(ds["weight"])
    obs_leads = 10
    input_step=10
    output_step=10
    in_ch = 1
    out_ch = 1
    netname = ['ConvLSTM','AFNONet','DNNNet','AttU_Net','R2AttU_Net','UNet','ConvLSTM_UNet','CNN']
    
    model_id = 0
    train_loader, val_loader, test_loader = dataset(batch_size=args.bs, obs_leads=obs_leads, model=netname[model_id], dim=5, input_step=input_step, output_step=output_step)
    device = args.local_rank
    weights = torch.from_numpy(weights).to(device)
    if dist.get_rank() == 0:
        print(f"Using {device} device")
    model = SwinLSTM(img_size=(720,1440), patch_size=(4,4),
                        in_chans=1, embed_dim=12,
                        depths_downsample=[2,4], depths_upsample=[2,4],
                        num_heads=[4,8], window_size=1).to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True, broadcast_buffers=False,)
    #model = model_list[model_id](img_size=[80, 160], in_chans=input_step, out_chans=output_step, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)).to(device)
    if dist.get_rank() == 0:
        print(model)
        if not os.path.exists(f'./results'):
            os.mkdir(f'./results')
    
    loss_fn = EarthMSELoss(weights=weights).to(device)
    #loss_fn = MaskedMSELoss2(mask=mask)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
    #T_0=10, T_mult=2, eta_min=1e-5, last_epoch=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5*1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    modelpath = f'./results/{netname[model_id]}_model_epoch99.pth'
    model.load_state_dict(torch.load(modelpath))

    epochs = args.e
    train_loss_list, val_loss_list, test_loss_list = [], [], []
    train_ssim_list, val_ssim_list, test_ssim_list = [], [], []
    loss_state = 1e10
    loss_outpath = f'./results/{netname[model_id]}_loss.json'
    for t in range(epochs):
        if dist.get_rank() == 0:
            print(f"Epoch {t+1}\n-------------------------------")
        modelpath = f'./results/{netname[model_id]}_model_epoch{t}.pth'
        train_loss = train(train_loader, model, loss_fn, optimizer, device=device)
        _, val_loss, test_loss, train_ssim, val_ssim, test_ssim  = test(train_loss=train_loss, train_loader = train_loader,
                                                        val_loader = val_loader, 
                                                        test_loader = test_loader, 
                                                        model = model, 
                                                        loss_fn = loss_fn,
                                                        device=device)
        #lr_sched.step()

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)

        #train_ssim_list.append(float(train_ssim.cpu()))
        #val_ssim_list.append(float(val_ssim.cpu()))
        #test_ssim_list.append(float(test_ssim.cpu()))

        if dist.get_rank() == 0:
            # loss results save!
            model_dic = {'train_loss' : train_loss_list,
                         'val_loss' : val_loss_list,
                        'test_loss' : test_loss_list,
                        'train_ssim' : train_ssim_list,
                         'val_ssim' : val_ssim_list,
                        'test_ssim' : test_ssim_list,
                        }

            torch.save(model.state_dict(), modelpath)

            with open(loss_outpath, 'w') as f:
            
                json.dump(model_dic, f, indent = 4, sort_keys = True)
                f.close()


            if loss_state>val_loss_list[-1]:
                #torch.save(model.state_dict(), modelpath)
                print(f"Saved PyTorch Model State to {modelpath},loss from {loss_state:>.7f} to {val_loss:>.7f}")
                loss_state = val_loss_list[-1]
    print("Done!")
