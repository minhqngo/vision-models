import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import stochastic_depth


class LayerNorm(nn.Module):
    """
    Support two tensor format: channels_last (N, H, W, C) and channels_first (N, C, H, W)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXTBlock(nn.Module):
    """
    DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    """

    def __init__(self, dim, drop_rate=0., layer_scale=1e-6):
        super(ConvNeXTBlock, self).__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        if layer_scale > 0:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma = None

        self.drop_rate = drop_rate

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + stochastic_depth(x, self.drop_rate, mode='batch', training=self.training)
        return x


class ConvNeXTLayer(nn.Module):
    """
    DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    """

    def __init__(self, dim, depth, drop_rates, layer_scale=1e-6):
        super(ConvNeXTLayer, self).__init__()
        assert depth == len(drop_rates), "depth must be equal size of drop_rates list"

        blocks = []
        for i in range(depth):
            blocks.append(ConvNeXTBlock(dim=dim, drop_rate=drop_rates[i], layer_scale=layer_scale))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


class ConvNeXT(nn.Module):
    def __init__(
        self,
        n_classes=200,
        patch_size=4,
        depths=[2, 2, 6, 2],
        dims=[40, 80, 160, 320],
        drop_rate=0.,
        layer_scale=1e-6
    ):
        super(ConvNeXT, self).__init__()

        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=patch_size, stride=patch_size),
                LayerNorm(dims[0], eps=1e-6, data_format='channels_first')
            )
        ])
        for i in range(len(dims) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
                )
            )

        drop_rates = [x.item() for x in torch.linspace(0, drop_rate, sum(depths))]
        self.stage_layers = nn.ModuleList([])
        for i, dim in enumerate(dims):
            dr = drop_rates[sum(depths[:i]):sum(depths[:i + 1])]
            self.stage_layers.append(
                ConvNeXTLayer(dim=dim, depth=depths[i], drop_rates=dr, layer_scale=layer_scale)
            )

        self.cls = nn.Sequential(
            LayerNorm(dims[-1], eps=1e-6),
            nn.Linear(dims[-1], n_classes)
        )

    def forward(self, x):
        for downsample_layer, stage_layer in zip(self.downsample_layers, self.stage_layers):
            x = downsample_layer(x)
            x = stage_layer(x)

        return self.cls(x.mean(dim=(-2, -1)))


def convnext_atto(**kwargs):
    return ConvNeXT(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)


def convnext_femto(**kwargs):
    return ConvNeXT(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)


def convnext_pico(**kwargs):
    return ConvNeXT(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)


if __name__ == '__main__':
    net = convnext_pico(n_classes=200)
    inp = torch.randn((1, 3, 64, 64))
    oup = net(inp)
    print(oup.size())
    # torch.save(net.state_dict(), 'convnext.pth')
