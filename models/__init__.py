from prettytable import PrettyTable

from .convnext import convnext_atto, convnext_femto, convnext_pico
from .mobilenetv3 import (mobilenet_v3_small, mobilenet_v3_small_50, mobilenet_v3_small_75,
                          mobilenet_v3_large, mobilenet_v3_large_50, mobilenet_v3_large_75)
from .resnet import resnet18, resnet34, resnet50
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
    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2
}


def create_model(model_name, **kwargs):
    if model_name not in dispatcher:
        raise NotImplementedError(f"Model {model_name} is not implemented.")

    return dispatcher[model_name](**kwargs)


def model_summary(model):
    print("model_summary")
    table = PrettyTable(["Modules", "Parameters", "Trainable parameters"])
    total_params = 0
    total_trainable_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        if parameter.requires_grad:
            trainable_params = params
            total_trainable_params += trainable_params
        else:
            trainable_params = 0
        table.add_row([name, params, trainable_params])
    print(table)
    print(f"Total params: {total_params}")
    print(f"Total trainable params: {total_trainable_params}")
    print()
