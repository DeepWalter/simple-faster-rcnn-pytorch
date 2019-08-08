import torch.nn as nn
import torch.nn.functional as F


class _RPN(nn.Module):
    """Region proposal networks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of output channels of the first convolutional layer.
    k : int
        Number of maximum region proposals at each location.
    """

    def __init__(self, in_channels=512, mid_channels=512, k=9):
        super(_RPN, self).__init__()

        self.k = k

        self.conv = nn.Conv2d(in_channels, mid_channels,
                              kernel_size=3, padding=1)
        self.cls = nn.Conv2d(mid_channels, 2 * k, kernel_size=1)
        self.reg = nn.Conv2d(mid_channels, 4 * k, kernel_size=1)

        # TODO: weight init.

    def forward(self, x):
        # x : (N, C, H, W)
        N, _, H, W = x.size()
        h = F.relu(self.conv(x), inplace=True)

        rpn_cls = self.cls(h)
        rpn_cls = F.softmax(rpn_cls.view(N, self.k, 2, H, W),
                            dim=2).view(N, -1, H, W)
        rpn_locs = self.reg(h)

        return rpn_cls, rpn_locs
