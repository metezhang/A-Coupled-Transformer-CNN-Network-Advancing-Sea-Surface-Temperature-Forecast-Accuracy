import torch
import sys
from torch import nn
from collections import OrderedDict

import torch
import torch.nn as nn

class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      2 * self.num_features, self.filter_size, 1,
                      self.padding))
            #nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, self.filter_size, 1, self.padding))
            #nn.GroupNorm(self.num_features // 32, self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1], device=self.conv1[0].weight.device)
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1], device=self.conv1[0].weight.device)
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            # zgate, rgate = gates.chunk(2, 1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev),
                                   1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features, dropout=0.3):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding))
            #nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))
        self.conv_last = nn.Conv2d(self.num_features * 4, self.num_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout = dropout 


    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1], device=self.conv_last.weight.device)
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1], device=self.conv_last.weight.device)
        else:
            hx, cx = hidden_state
        output_inner = []

        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1], device=self.conv_last.weight.device)
            else:
                x = inputs[index, ...]
            _ones = torch.ones_like(x)

            dp_mask = nn.functional.dropout(_ones, p=self.dropout)
            x = x*dp_mask

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        elif 'dropout' in layer_name:
            layer = nn.Dropout2d(p=v[0])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()  # [52,4,1,20,100]
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))  # [2016,1,20,100]
        inputs = subnet(inputs)  # [2016,16,20,100]
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        #print('encoder shape:',inputs.shape)  # [52,4,16,20,100] ; batch size=4
        outputs_stage, state_stage = rnn(inputs, None, seq_len=10)
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to 52,B,1,20,100
        hidden_states = []
        #logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)


class Forecaster(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks-index), rnn)
            setattr(self, 'stage' + str(self.blocks-index), make_layers(params))

    def forward_by_stage(self, input, state, subnet, rnn):
        input, state_stage = rnn(input, state, seq_len=10)
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        #print('decoder shape:', input.shape)
        return input

        # input: 5D S*B*I*H*W

    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage2'),
                                      getattr(self, 'rnn2'))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i-1], getattr(self, 'stage' + str(i)),
                                                       getattr(self, 'rnn' + str(i)))
        input = input.permute(1, 0, 2, 3, 4)
        return input

class EF(nn.Module):

    def __init__(self, encoder, forecaster):
        super().__init__()
        self.encoder = encoder
        self.forecaster = forecaster
        self.MSE_criterion = torch.nn.MSELoss()

    def forward(self, input, labels):
        state = self.encoder(input)
        output = self.forecaster(state)
        loss = self.MSE_criterion(output, labels)
        return output, loss

class EF_Conv(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.net = args
        #self.MSE_criterion = torch.nn.MSELoss()
        if self.net == 'convgru':
            self.encoder_params = convgru_encoder_params
            self.decoder_params = convgru_decoder_params
        elif self.net == 'convlstm':
            self.encoder_params = convlstm_encoder_params
            self.decoder_params = convlstm_decoder_params
        self.encoder = Encoder(self.encoder_params[0], self.encoder_params[1])
        self.forecaster = Forecaster(self.decoder_params[0], self.decoder_params[1])

    def forward(self, inputs):
        state = self.encoder(inputs)
        output = self.forecaster(state)
        #loss = self.MSE_criterion(output, labels)
        return output



convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 16, 3, 2, 1]}),
        OrderedDict({'conv2_leaky_1': [16, 16, 3, 2, 1]})
    ],

    [
        CLSTM_cell(shape=(360,720),input_channels=16, filter_size=3, num_features=16),
        CLSTM_cell(shape=(180,360),input_channels=16, filter_size=3, num_features=16)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [16, 16, 4, 2, 1]}),
        OrderedDict({
            'deconv2_leaky_1': [16, 16, 4, 2, 1],
            'conv3_leaky_1': [16, 1, 1, 1, 0]
        })
        # OrderedDict({
        #     'conv2_leaky_1': [16, 16, 3, 1, 1],
        #     'conv3_leaky_1': [16, 1, 1, 1, 0]
        # }),
    ],

    [
        CLSTM_cell(shape=(180,360),input_channels=16, filter_size=3, num_features=16),
        CLSTM_cell(shape=(360,720),input_channels=16, filter_size=3, num_features=16)
    ]
]

convgru_encoder_params = [
    [
        OrderedDict({
            'conv1_leaky_1': [1, 16, 3, 2, 1],
            'dropout1':[0.2]
        }),
        OrderedDict({
            'conv2_leaky_1': [16, 16, 3, 2, 1],
            'dropout2':[0.2]
                     }),

    ],

    [
        CGRU_cell(shape=(360,720), input_channels=16, filter_size=5, num_features=16),
        CGRU_cell(shape=(180,360), input_channels=16, filter_size=5, num_features=16)
    ]
]

convgru_decoder_params = [
    [
        OrderedDict({
            'deconv1_leaky_1': [16, 16, 4, 2, 1],
            'dropout2':[0.2]
        }),

        OrderedDict({
            'deconv2_leaky_1': [16, 16, 4, 2, 1],
             'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
    ],

    [
        CGRU_cell(shape=(180,360), input_channels=16, filter_size=5, num_features=16),
        CGRU_cell(shape=(360,720), input_channels=16, filter_size=5, num_features=16)
    ]
]
from torch.autograd import Variable

#x = torch.rand((2, 10, 1, 160, 160))
#y = torch.rand((2, 15, 1, 160, 160))
#net=EF_Conv('convlstm')
#output, loss = net(x,y)
#print(loss)
#print(net)
