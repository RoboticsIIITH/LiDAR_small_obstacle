import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
        dropout = 0
    ):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()

        for i in range(depth):
            self.down_path.append(
                Conv(prev_channels, 2 ** (wf + i))
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm,dropout)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv1d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x,*args):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool1d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm,dropout):
        super(UNetUpBlock, self).__init__()

        if up_mode == 'upconv':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_size, out_size, kernel_size=(3,3),stride=2,padding=1,output_padding=1),
                nn.LeakyReLU())

        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = nn.Sequential(nn.Conv1d(in_size,out_size,kernel_size=3,padding=1),
                                        nn.LeakyReLU())

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat((up,bridge),dim=1)
        out = self.conv_block(out)
        return out


class ContextNet(nn.Module):

    def __init__(self,inp_size,out_size):
        super(ContextNet, self).__init__()
        self.conv_1 = nn.Conv2d(inp_size,64,kernel_size=(3,7),stride=1,padding=(1,3))
        self.conv_2 = nn.Conv2d(64,32,kernel_size=(3,7),stride=1,padding=(1,3))
        self.conv_3 = nn.Conv2d(32,out_size,kernel_size=(3,7),stride=1,padding=(1,3))

    def forward(self, x):
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2(x))
        x = self.conv_3(x)
        return x


if __name__=="__main__":
    import numpy as np
    model = ContextNet(5,2)
    inp = np.random.randn(16,5,7,1280)
    inp = torch.from_numpy(inp).float()
    out = model.forward(inp)
    print(out.shape)
    print(model)