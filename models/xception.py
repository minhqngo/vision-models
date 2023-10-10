from torch import nn


class SeparableConv2D(nn.Module):

    def __init__(self, inp, oup, kernel, **kwargs):
        super(SeparableConv2D, self).__init__()

        # depthwise
        self.dw = nn.Conv2d(inp, oup, kernel_size=kernel, groups=inp, bias=False, **kwargs)
        # pointwise
        self.pw = nn.Conv2d(inp, oup, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x
