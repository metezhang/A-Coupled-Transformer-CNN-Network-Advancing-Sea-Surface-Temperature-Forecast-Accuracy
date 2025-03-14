__author__ = 'yunbo'

import torch
import torch.nn as nn
#from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell


import torch
import torch.nn as nn

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                #nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                #nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                #nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                #nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = 1
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            #in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            in_channel = 1 if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        #frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        #mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        
        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(frames.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(frames.device)

        for t in range(self.configs.total_length - 1):
            # # reverse schedule sampling
            # if self.configs.reverse_scheduled_sampling == 1:
            #     if t == 0:
            #         net = frames[:, t]
            #     else:
            #         net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            # else:
            #     if t < self.configs.input_length:
            #         net = frames[:, t]
            #     else:
            #         net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
            #               (1 - mask_true[:, t - self.configs.input_length]) * x_gen
            
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                net = x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
            #print(next_frames[-1].shape)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        #next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        next_frames = torch.stack(next_frames[-(self.configs.total_length - self.configs.input_length):], dim=1)
        #loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        return next_frames
    
class ModelConfig:
    def __init__(self, model_name='predrnn', pretrained_model='', num_hidden='64', img_channel=1, img_width=(64),
                 filter_size=5, stride=1, patch_size=4, layer_norm=1, decouple_beta=0.1,
                 total_length=3, input_length=1,):
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

configs = ModelConfig()

