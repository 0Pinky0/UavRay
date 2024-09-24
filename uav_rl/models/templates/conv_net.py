from typing import Optional, Sequence, Type

import torch
from torch import nn

from .activations import get_activation_class


class SquashDims(nn.Module):
    def __init__(self, ndims_in: int = 3):
        super().__init__()
        self.ndims_in = ndims_in

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.flatten(-self.ndims_in, -1)
        return value


class ConvNet(nn.Sequential):
    def __init__(self,
                 in_channels: Optional[int] = None,
                 num_channels: Sequence[int] = (),
                 kernel_sizes: Sequence[int] | int = 3,
                 strides: Sequence[int] | int = 1,
                 paddings: Sequence[int] | int = 0,
                 activation_class: Optional[Type[nn.Module] | str] = nn.ReLU,
                 activation_kwargs: Optional[dict] = None,
                 norm_class: Optional[Type[nn.Module]] = None,
                 norm_kwargs: Optional[dict] = None,
                 squash_last_layer: bool = True):
        if isinstance(activation_class, str):
            activation_class = get_activation_class(activation_class)
        if not activation_kwargs:
            activation_kwargs = {}
        depth = len(num_channels)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * depth
        if isinstance(strides, int):
            strides = [strides] * depth
        if isinstance(paddings, int):
            paddings = [paddings] * depth
        assert len(kernel_sizes) == depth and len(strides) == depth and len(paddings) == depth, \
            "Not all parameters of conv2d are of the same length"
        layers = []
        _in = in_channels
        for _out, _kernel, _stride, _padding in zip(num_channels, kernel_sizes, strides, paddings):
            if _in:
                layers.append(nn.Conv2d(
                    _in,
                    _out,
                    kernel_size=_kernel,
                    stride=_stride,
                    padding=_padding,
                ))
            else:
                layers.append(nn.LazyConv2d(
                    _out,
                    kernel_size=_kernel,
                    stride=_stride,
                    padding=_padding,
                ))
            if activation_class:
                layers.append(activation_class(**activation_kwargs))
            if norm_class:
                layers.append(norm_class(**norm_kwargs))
            _in = _out
        if squash_last_layer:
            layers.append(SquashDims())
        super().__init__(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batch, C, L, W = inputs.shape
        if len(batch) > 1:
            inputs = inputs.flatten(0, len(batch) - 1)
        out = super(ConvNet, self).forward(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        return out
