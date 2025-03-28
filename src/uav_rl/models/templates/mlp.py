from typing import Optional, Sequence, Type, Tuple

import torch
from torch import nn

from .activations import get_activation_class


class MLP(nn.Sequential):
    def __init__(self,
                 out_features: int,
                 in_features: Optional[int] = None,
                 num_cells: Sequence[int] = (),
                 activation_class: Optional[Type[nn.Module] | str] = nn.ReLU,
                 activation_kwargs: Optional[dict] = None,
                 activate_last_layer: bool = False):
        if isinstance(activation_class, str):
            activation_class = get_activation_class(activation_class)
        if not activation_kwargs:
            activation_kwargs = {}
        self.out_features = out_features
        layers = []
        _in = in_features
        for _out in num_cells:
            if _in:
                layers.append(nn.Linear(_in, _out))
            else:
                layers.append(nn.LazyLinear(_out))
            if activation_class:
                layers.append(activation_class(**activation_kwargs))
            _in = _out
        if _in:
            layers.append(nn.Linear(_in, out_features))
        else:
            layers.append(nn.LazyLinear(out_features))
        if activate_last_layer:
            layers.append(activation_class(**activation_kwargs))
        super().__init__(*layers)

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            inputs = (torch.cat([*inputs], -1),)
        out = super().forward(*inputs)
        # if not isinstance(self.out_features, Number):
        #     out = out.view(*out.shape[:-1], *self.out_features)
        return out
