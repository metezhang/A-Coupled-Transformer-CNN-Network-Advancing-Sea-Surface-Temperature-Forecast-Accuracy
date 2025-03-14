import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset,random_split
import numpy as np
import argparse
import os
import xarray as xr
import math
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, dropout=0):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.dropout = dropout

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):

        _ones = torch.ones_like(input_tensor)
        dp_mask = nn.functional.dropout(_ones, p=self.dropout)
        input_tensor = input_tensor*dp_mask

        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False, dropout=0.2):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.dropout = dropout

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          dropout=self.dropout))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()

        n1 = 8 
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out

class TimeDistributed(nn.Module):
    def __init__(self, layer):
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        outputs = []
        for t in range(x.size(1)):
            xt = x[:, t, :, :, :]
            output = self.layer(xt)
            outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs

class ConvLSTMNetwork(nn.Module):
    def __init__(self,input_step=10, output_step=10,in_ch=1, out_ch=1, dropout=0.2, keep_in_memory=False):
        super().__init__()
        self.keep_in_memory = keep_in_memory
        self.input_step = input_step
        self.output_step = output_step
        self.encoder = ConvLSTM(input_dim=in_ch,
                 hidden_dim=[8, 8,],
                 kernel_size=(3, 3),
                 num_layers=2,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False,
                 dropout=dropout,
                 )

        self.decoder = ConvLSTM(input_dim=8,
            hidden_dim=[8, 8,],
            kernel_size=(3, 3),
            num_layers=2,
            batch_first=True,
            bias=True,
            return_all_layers=False,
            dropout=dropout,
            )

        self.Tconv = TimeDistributed(nn.Conv2d(8, out_ch, 3, stride=1, padding=(1, 1))) 
        
    def forward(self, x):
        # [sample, time, ch, lat, lon] of x_shape
        _,x = self.encoder(x)
        x = x[0][0] # last state
        x = torch.unsqueeze(x, dim=1)
        x = torch.tile(x,(1,self.output_step,1,1,1))
        #x,_ = self.encoder(x)
        #x = x[0]
        x,_ = self.decoder(x)

        x = x[0]
        x = self.Tconv(x)
        return x

class ConvLSTMNetwork2(nn.Module):
    def __init__(self,input_step=10, output_step=10, keep_in_memory=False):
        super().__init__()
        self.keep_in_memory = keep_in_memory
        self.input_step = input_step
        self.output_step = output_step
        self.encoder1 = ConvLSTM(input_dim=1,
                 hidden_dim=[64],
                 kernel_size=(3, 3),
                 num_layers=1,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False,
                 )
        self.encoder2 = ConvLSTM(input_dim=64,
                 hidden_dim=[64],
                 kernel_size=(3, 3),
                 num_layers=1,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False,
                 )
        self.decoder1 = ConvLSTM(input_dim=64,
            hidden_dim=[64],
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False,
            )
        self.decoder2 = ConvLSTM(input_dim=64,
                 hidden_dim=[64],
                 kernel_size=(3, 3),
                 num_layers=1,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False,
                 )
        self.decoder3 = ConvLSTM(input_dim=64,
                 hidden_dim=[1],
                 kernel_size=(3, 3),
                 num_layers=1,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False,
                 )
        
    def forward(self, x):
        # [sample, time, ch, lat, lon] of x_shape
        x,_ = self.encoder1(x)
        x = x[0]
        x = nn.Dropout(0.3)(x)
        _,x = self.encoder2(x)
        x = x[0][0]
        x = nn.Dropout(0.3)(x)

        x = torch.unsqueeze(x, dim=1)
        x = torch.tile(x,(1,self.output_step,1,1,1))
        x,_ = self.decoder1(x)
        x = x[0]
        x = nn.Dropout(0.3)(x)
        x,_ = self.decoder2(x)
        x = x[0]
        x = nn.Dropout(0.3)(x)
        x,_ = self.decoder3(x)
        x = x[0]
        return x
class ConvLSTM_UNet_Network(nn.Module):
    def __init__(self,input_step=10, output_step=10, keep_in_memory=False):
        super().__init__()
        self.keep_in_memory = keep_in_memory
        self.input_step = input_step
        self.output_step = output_step
        self.encoder = ConvLSTM(input_dim=1,
                 hidden_dim=[ 64,],
                 kernel_size=(3, 3),
                 num_layers=1,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False,
                 )
        self.decoder = UNet(64,self.output_step)

    def forward(self, x):
        # [sample, time, ch, lat, lon] of x_shape
        _,x = self.encoder(x)
        x = x[0][0] # last state
        x = self.decoder(x)
        return x

class SegNet(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))

        #x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=1,return_indices=True)
        x1p, id1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, 
			dilation=1, return_indices=True, ceil_mode=False)(x12) 

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        #x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=1,return_indices=True)
        x2p, id2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0,
                        dilation=1, return_indices=True, ceil_mode=False)(x22)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        #x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=1,return_indices=True)
        x3p, id3 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0,
                        dilation=1, return_indices=True, ceil_mode=False)(x33)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        #x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=1,return_indices=True)
        x4p, id4 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0,
                        dilation=1, return_indices=True, ceil_mode=False)(x43)
 

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        #x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=1,return_indices=True)
        x5p, id5 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0,
                        dilation=1, return_indices=True, ceil_mode=False)(x53)
 


        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=1)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=1)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=1)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=1)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=1)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        x11d = self.sigmoid(x11d)
        return x11d

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(model_path).state_dict() # load the weigths
        # for name in th:
            # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)

class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """
    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out


class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

      # out = self.active(out)

        return out


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out


class R2AttU_Net(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """
    def __init__(self, in_ch=3, out_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNN_block(in_ch, filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)
        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out

#For nested 3 channels are required

class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

#from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
#from utils.params import get_args
from torch.utils.checkpoint import checkpoint_sequential


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = nn.AdaptiveAvgPool1d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaptiveFourierNeuralOperator(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.hidden_size = dim
        self.h = h
        self.w = w

        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()

        if True:
            self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        else:
            self.bias = None

        self.softshrink = 0

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape

        if self.bias:
            bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            bias = torch.zeros(x.shape, device=x.device)

        x = x.reshape(B, self.h, self.w, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        x_real = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0], inplace=True)
        x_imag = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1], inplace=True)
        x_real = self.multiply(x_real, self.w2[0]) - self.multiply(x_imag, self.w2[1]) + self.b2[0]
        x_imag = self.multiply(x_real, self.w2[1]) + self.multiply(x_imag, self.w2[0]) + self.b2[1]

        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)

        return x + bias


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AdaptiveFourierNeuralOperator(dim, h=h, w=w)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.double_skip = True

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x += residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x += residual
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=None, patch_size=8, in_chans=13, embed_dim=768):
        super().__init__()

        if img_size is None:
            raise KeyError('img is None')

        patch_size = to_2tuple(patch_size)

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class AFNONet(nn.Module):
    def __init__(self, img_size=None, patch_size=8, in_chans=20, out_chans=20, embed_dim=768, depth=12, mlp_ratio=4.,
                 uniform_drop=False, drop_rate=0., drop_path_rate=0., norm_layer=None, dropcls=0):
        super().__init__()

        if img_size is None:
            img_size = [720, 1440]

        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size

        if uniform_drop:
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, h=self.h, w=self.w) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        # self.num_features = out_chans * img_size[0] * img_size[1]
        # self.representation_size = self.num_features * 8
        # self.pre_logits = nn.Sequential(OrderedDict([
        #     ('fc', nn.Linear(embed_dim, self.representation_size)),
        #     ('act', nn.Tanh())
        # ]))
        self.pre_logits = nn.Sequential(OrderedDict([
            ('conv1', nn.ConvTranspose2d(embed_dim, out_chans*16, kernel_size=(2, 2), stride=(2, 2))),
            ('act1', nn.Tanh()),
            ('conv2', nn.ConvTranspose2d(out_chans*16, out_chans*4, kernel_size=(2, 2), stride=(2, 2))),
            ('act2', nn.Tanh())
        ]))

        # Generator head
        # self.head = nn.Linear(self.representation_size, self.num_features)
        self.head = nn.ConvTranspose2d(out_chans*4, out_chans, kernel_size=(2, 2), stride=(2, 2))

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x += self.pos_embed
        x = self.pos_drop(x)

        if not True:
            for blk in self.blocks:
                x = blk(x)
        else:
            x = checkpoint_sequential(self.blocks, 4, x)

        x = self.norm(x).transpose(1, 2)
        x = torch.reshape(x, [-1, self.embed_dim, self.h, self.w])
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.pre_logits(x)
        x = self.head(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

#class conv_block(nn.Module):
#    """
#    Convolution Block 
#    """
#    def __init__(self, in_ch, out_ch):
#        super(conv_block, self).__init__()
#        
#        self.conv = nn.Sequential(
#            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=1, bias=True),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=2, stride=2))
#        self.conv1 = nn.Conv2d(in_ch+out_ch, 8,kernel_size=5, stride=1, padding=2, bias=True)
#        self.conv2 = nn.Conv2d(8, 16,kernel_size=3, stride=1, padding=1, bias=True)
#        self.conv3 = nn.Conv2d(16, 32,kernel_size=3, stride=1, padding=1, bias=True)
#        self.conv4 = nn.Conv2d(32 +in_ch+out_ch, out_ch,kernel_size=5, stride=1, padding=2, bias=True)
#            
#
#    def forward(self, x):
#
#        x1, x2 = x[0], x[1] # x1 shape [sample, x_step, lat, lon] , x2 shape [sample, out_ch, lat, lon]
#        x = torch.cat(x, axis=1) # shape [sample, x_step+1, lat, lon]
#        x = self.conv1(x)
#        x = torch.tanh(x)
#        x = self.conv2(x)
#        x = torch.tanh(x)
#        x = self.conv3(x)
#        x = torch.tanh(x)
#        x = torch.cat([x, x1, x2], axis=1) # shape [sample, 32 + x_step+1, lat, lon]
#        x = self.conv4(x)
#        return x
#
#class conv_block2(nn.Module):
#    """
#    Convolution Block 
#    """
#    def __init__(self, in_ch, out_ch):
#        super(conv_block2, self).__init__()
#        self.conv = nn.Sequential(
#            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=1, bias=True),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=2, stride=2))
#        
#        self.conv1 = nn.Conv2d(in_ch, 8,kernel_size=5, stride=1, padding=2, bias=True)
#        self.conv2 = nn.Conv2d(8, 16,kernel_size=3, stride=1, padding=1, bias=True)
#        self.conv3 = nn.Conv2d(16, 32,kernel_size=3, stride=1, padding=1, bias=True)
#        self.conv4 = nn.Conv2d(32 +in_ch, out_ch,kernel_size=5, stride=1, padding=2, bias=True)
#            
#
#    def forward(self, x):
#        # x shape [sample, x_step+1, lat, lon]
#        x_input = x
#        x = self.conv1(x)
#        x = torch.tanh(x)
#        x = self.conv2(x)
#        x = torch.tanh(x)
#        x = self.conv3(x)
#        x = torch.tanh(x)
#        x = torch.cat([x, x_input], axis=1) # shape [sample, 32 + x_step+1, lat, lon]
#        x = self.conv4(x)
#        return x
#
#class DNN_Net(nn.Module):
#    def __init__(self, in_ch, out_ch):
#        super(DNN_Net, self).__init__()
#        self.conv_block1 = conv_block(in_ch, out_ch)
#        self.conv_block2 = conv_block(in_ch, out_ch)
#        self.conv_block3 = conv_block(in_ch, out_ch)
#        self.conv_block4 = conv_block2(in_ch, out_ch)
#        self.pool = nn.AvgPool2d(kernel_size=(2,2))
#        self.up = nn.Upsample(scale_factor=2)
#    
#    def forward(self, x):
#        
#        down1 = self.pool(x)
#        down2 = self.pool(down1)
#        down3 = self.pool(down2)
#
#        x4 = self.conv_block4(down3)
#        x4 = self.up(x4)
#
#        x3 = self.conv_block3([down2, x4])
#        x3 = self.up(x3)
#
#        x2 = self.conv_block2([down1, x3])
#        x2 = self.up(x3)
#
#        x = self.conv_block1([x, x2])
#        return x
