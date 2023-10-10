import math
import torch
from torch import nn

__all__ = [
    'EfficientNet',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2'
]


def round_channels(channel, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor

    new_channel = max(min_value, (channel + divisor // 2) // divisor * divisor)
    if new_channel < 0.9 * channel:
        new_channel += divisor
    return int(new_channel)


def round_repeats(rep):
    return int(math.ceil(rep))


def drop_path(x, p, training=False):
    if p > 0 and training:
        keep_prob = 1 - p

        device = x.device
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(device)
        x.div_(keep_prob)
        x.mul_(mask)

    return x


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * nn.Sigmoid()(x)


def conv_1x1_bn(inp, oup, activation=Swish):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup, eps=1e-3, momentum=0.01),
        activation()
    )


def conv_3x3_bn(inp, oup, stride, activation=Swish):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup, eps=1e-3, momentum=0.01),
        activation()
    )


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.activation1 = Swish()
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.activation2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.activation1(self.se_reduce(y))
        y = self.activation2(self.se_expand(y))
        y = x * y
        return y


class MBConvBlock(nn.Module):
    def __init__(self, inp, oup, kernel, stride, expand_ratio, se_ratio, drop_rate):
        super(MBConvBlock, self).__init__()

        expand_channels = inp * expand_ratio
        self.residual_connection = (stride == 1 and inp == oup)
        self.drop_rate = drop_rate

        layers = []

        if expand_ratio != 1.:
            # expansion
            pw_expand = nn.Sequential(
                nn.Conv2d(inp, expand_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(expand_channels, eps=1e-3, momentum=0.01),
                Swish()
            )
            layers.append(pw_expand)

        # depthwise
        dw = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels,
                      kernel_size=kernel, stride=stride,
                      padding=kernel // 2, groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels, eps=1e-3, momentum=0.01),
            Swish()
        )
        layers.append(dw)

        if se_ratio != 0.0:
            # squeeze and excite
            squeeze_excite = SqueezeAndExcite(expand_channels, inp, se_ratio)
            layers.append(squeeze_excite)

        # projection
        pw_project = nn.Sequential(
            nn.Conv2d(expand_channels, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup, eps=1e-3, momentum=0.01)
        )
        layers.append(pw_project)

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            return x + drop_path(self.conv(x), self.drop_rate, self.training)
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    config = [
        # [inp, oup, kernel, stride, expand_ratio, se_ratio, repeats]
        [32, 16, 3, 1, 1, 0.25, 1],
        [16, 24, 3, 2, 6, 0.25, 2],
        [24, 40, 5, 2, 6, 0.25, 2],
        [40, 80, 3, 2, 6, 0.25, 3],
        [80, 112, 5, 1, 6, 0.25, 3],
        [112, 192, 5, 2, 6, 0.25, 4],
        [192, 320, 3, 1, 6, 0.25, 1]
    ]

    def __init__(self, width_coef, depth_coef, n_classes=1000, drop_rate=0.,
                 stem_channels=32, feature_size=1280, drop_conn_rate=0.2):
        super(EfficientNet, self).__init__()

        # scaling width
        if width_coef != 1.:
            stem_channels = round_channels(stem_channels * width_coef)
            for cfg in self.config:
                cfg[0] = round_channels(cfg[0] * width_coef)
                cfg[1] = round_channels(cfg[1] * width_coef)

        # scaling depth
        if depth_coef != 1.:
            for cfg in self.config:
                cfg[6] = round_repeats(cfg[6] * depth_coef)

        self.stem = conv_3x3_bn(3, stem_channels, 2)

        total_blocks = sum([cfg[6] for cfg in self.config])
        blocks = []
        for [inp, oup, kernel, stride, expand_ratio, se_ratio, repeats] in self.config:
            p = drop_conn_rate * len(blocks) / total_blocks
            blocks.append(MBConvBlock(inp, oup, kernel, stride, expand_ratio, se_ratio, p))
            for _ in range(repeats - 1):
                p = drop_conn_rate * len(blocks) / total_blocks
                blocks.append(MBConvBlock(oup, oup, kernel, 1, expand_ratio, se_ratio, p))
        self.blocks = nn.Sequential(*blocks)

        self.last_conv = conv_1x1_bn(self.config[-1][1], feature_size)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feature_size, n_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.last_conv(x)
        x = torch.mean(x, (2, 3))
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def efficientnet_b0(**kwargs):
    return EfficientNet(width_coef=1., depth_coef=1., **kwargs)


def efficientnet_b1(**kwargs):
    return EfficientNet(width_coef=1., depth_coef=1.1, **kwargs)


def efficientnet_b2(**kwargs):
    return EfficientNet(width_coef=1.1, depth_coef=1.2, **kwargs)


if __name__ == '__main__':
    net = efficientnet_b2(n_classes=200)
    inp = torch.randn((1, 3, 64, 64))
    oup = net(inp)
    print(oup.size())
    # torch.save(net.state_dict(), 'efficientnet.pth')
