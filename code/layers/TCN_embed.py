import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
class GTCN(nn.Module):
    def __init__(self, device, num_nodes, in_dim,out_dim,dropout=0.3,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=3,blocks=4,layers=2):
        super(GTCN, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        for b in range(blocks):
            new_dilation = 1
            for i in range(layers):
                # 计算 same padding
                padding = (kernel_size - 1) * new_dilation // 2

                self.filter_convs.append(nn.Conv2d(
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    kernel_size=(1, kernel_size),
                    dilation=(1, new_dilation),
                    padding=(0, padding)  # 关键：添加 padding
                ))

                self.gate_convs.append(nn.Conv2d(  # 注意：原代码这里用了 Conv1d，应统一为 Conv2d！
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    kernel_size=(1, kernel_size),
                    dilation=(1, new_dilation),
                    padding=(0, padding)
                ))

                # residual 和 skip 保持 1x1，无需 padding
                self.residual_convs.append(nn.Conv2d(
                    in_channels=dilation_channels,
                    out_channels=residual_channels,
                    kernel_size=(1, 1)
                ))
                self.skip_convs.append(nn.Conv2d(
                    in_channels=dilation_channels,
                    out_channels=skip_channels,
                    kernel_size=(1, 1)
                ))

                self.bn.append(nn.BatchNorm2d(residual_channels))

                new_dilation *= 2


        self.end_conv_1 = nn.Conv2d(in_channels=residual_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)


    def forward(self, input):#
        #input_mark=torch.cat([input,mark],1)
        x = self.start_conv(input)
        skip = 0
        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip
            except:
                skip = 0
            skip = s + skip
            x = self.residual_convs[i](x)
            x = x + residual
            x = self.bn[i](x)

        x = F.relu(x)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x