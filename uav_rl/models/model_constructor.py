from typing import Any, Sequence

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from models.templates import (
    MLP,
    ConvNet,
    ResNetBlock,
    ImpalaNet,
)
import torch
from torch import nn

model_mapping = {
    'mlp': MLP,
    'conv_net': ConvNet,
    'resnet_block': ResNetBlock,
    'impala_net': ImpalaNet,
    'rnn': nn.RNN,
    'lstm': nn.LSTM,
    'gru': nn.GRU,
    'transformer': nn.Transformer,
}
default_configs = {
    'mlp': {
        'in_features': None,
        'out_features': 2,
        'num_cells': (32, 32),
        'activation_class': nn.Tanh,
        'activation_kwargs': None,
        'activate_last_layer': False,
    },
    'conv_net': {
        'in_channels': 3,
        'num_channels': (16, 32, 64),
        'kernel_sizes': (3, 3, 3),
        'strides': (1, 1, 1),
        'paddings': (0, 0, 0),
        'activation_class': nn.Tanh,
        'activation_kwargs': None,
        'norm_class': None,
        'norm_kwargs': None,
        'squash_last_layer': True,
    },
    'resnet_block': {
        'num_ch': 3,
        'activation_class': nn.Tanh,
        'activation_kwargs': None,
    },
    'impala_net': {
        'channels': (16, 32, 32),
        'activation_class': nn.Tanh,
        'activation_kwargs': None,
    },
    'rnn': {
        'input_size': 32,
        'hidden_size': 256,
        'num_layers': 1,
        'nonlinearity': 'tanh',
        'bias': True,
        'batch_first': True,
    },
    'lstm': {
        'input_size': 32,
        'hidden_size': 256,
        'num_layers': 1,
        'nonlinearity': 'tanh',
        'bias': True,
        'batch_first': True,
    },
    'gru': {
        'input_size': 32,
        'hidden_size': 256,
        'num_layers': 1,
        'nonlinearity': 'tanh',
        'bias': True,
        'batch_first': True,
    },
    'transformer': {
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
    },
    # 'auto_encoder': {},
}


def get_constructed_model(model_configs: Sequence[dict[str, Any]]) -> TensorDictSequential:
    layers = []
    for model_config in model_configs:
        model = model_mapping[model_config['model_name']](**model_config['model_config'])
        model = TensorDictModule(model, model_config['in_keys'], model_config['out_keys'])
        layers.append(model)
    # model = nn.Sequential(*layers)
    model = TensorDictSequential(*layers)
    return model


if __name__ == '__main__':
    test_model_configs = (
        {
            'model_name': 'conv_net',
            'in_keys': ['raster'],
            'out_keys': ['logits'],
            'model_config': {
                'num_channels': (16, 32, 64),
                'kernel_sizes': 3,
                'activation_class': 'relu',
                'squash_last_layer': True,
            },
        },
        {
            'model_name': 'mlp',
            'in_keys': ['logits', 'observation'],
            'out_keys': ['logits'],
            'model_config': {
                'out_features': 1,
                'num_cells': (512, 256),
                'activation_class': 'relu',
                'activate_last_layer': False,
            },
        },
    )
    test_model = get_constructed_model(test_model_configs)
    # dummy_data = torch.zeros([32, 3, 16, 16])
    dummy_data = {
        'raster': torch.zeros([32, 3, 16, 16]),
        'observation': torch.zeros([32, 8])
    }
    dummy_data = TensorDict(dummy_data)
    out = test_model(dummy_data)
    print(test_model)
    for key, value in out.items():
        print(f'{key}: {value.shape}')
    # pass
