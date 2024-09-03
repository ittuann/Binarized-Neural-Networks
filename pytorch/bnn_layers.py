from torch import nn
from bnn_ops import Binarize


class BinarizeLinear(nn.Linear):
    def __init__(self, *args, is_binarized_activation=True, **kwargs):
        kwargs.setdefault("bias", False)
        super(BinarizeLinear, self).__init__(*args, **kwargs)
        self.is_binarized_activation = is_binarized_activation

    def forward(self, input):
        if self.is_binarized_activation:
            input_b = Binarize.apply(input)
        else:
            input_b = input
        weight_b = Binarize.apply(self.weight)
        # y = xA^T + b 其中x是输入数据，A是权重矩阵，b是偏置项(可选)
        # nn.functional.linear函数内部已经处理了权重矩阵的转置，因此用户不需要显式地转置权重矩阵。
        out = nn.functional.linear(input_b, weight_b)
        if self.bias:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *args, is_binarized_activation=True, **kwargs):
        kwargs.setdefault("bias", False)
        super(BinarizeConv2d, self).__init__(*args, **kwargs)
        self.is_binarized_activation = is_binarized_activation

    def forward(self, input):
        if self.is_binarized_activation:
            input_b = Binarize.apply(input)
        else:
            input_b = input
        weight_b = Binarize.apply(self.weight)

        out = nn.functional.conv2d(
            input_b,
            weight_b,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        if self.bias:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


def Binaryconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )
