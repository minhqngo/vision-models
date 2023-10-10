from torch import nn

__all__ = [
    'VGG',
    'vgg11',
    'vgg13'
]


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}


def make_conv_layers(layer_cfg, batch_norm=False):
    layers = []

    inp = 3
    for i, l in enumerate(layer_cfg):
        if l == 'M':
            if i + 1 < len(layer_cfg):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        else:
            oup = l
            layers.append(nn.Conv2d(inp, oup, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(oup))
            layers.append(nn.ReLU(inplace=True))
            inp = l

    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, feature_layers, n_classes):
        super(VGG, self).__init__()

        self.features = feature_layers

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, n_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def vgg11(**kwargs):
    feature_layers = make_conv_layers(cfg['vgg11'], batch_norm=True)
    return VGG(feature_layers, **kwargs)


def vgg13(**kwargs):
    feature_layers = make_conv_layers(cfg['vgg13'], batch_norm=True)
    return VGG(feature_layers, **kwargs)


if __name__ == '__main__':
    import torch

    net = vgg13(n_classes=200)
    inp = torch.randn((1, 3, 64, 64))
    oup = net(inp)
    print(oup.size())
    # torch.save(net.state_dict(), 'vgg.pth')
