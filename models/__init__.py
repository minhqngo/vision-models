from .convnext import convnext_atto, convnext_femto, convnext_pico
from .mobilenetv3 import (mobilenet_v3_small, mobilenet_v3_small_50, mobilenet_v3_small_75,
                          mobilenet_v3_large, mobilenet_v3_large_50, mobilenet_v3_large_75)
from .resnet import resnet18, resnet34, resnet50
from .vgg import vgg11, vgg13
from .efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2

dispatcher = {
    'convnext_atto': convnext_atto,
    'convnext_femto': convnext_femto,
    'convnext_pico': convnext_pico,
    'mobilenet_v3_small': mobilenet_v3_small,
    'mobilenet_v3_small_50': mobilenet_v3_small_50,
    'mobilenet_v3_small_75': mobilenet_v3_small_75,
    'mobilenet_v3_large': mobilenet_v3_large,
    'mobilenet_v3_large_50': mobilenet_v3_large_50,
    'mobilenet_v3_large_75': mobilenet_v3_large_75,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2
}


def create_model(model_name, **kwargs):
    if model_name not in dispatcher:
        raise NotImplementedError(f"Model {model_name} is not implemented.")

    return dispatcher[model_name](**kwargs)
