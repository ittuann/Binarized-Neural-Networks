"""eBNN层

Note:
    File   : ebnn_layers.py
    Author : Baiqi.Lu <ittuann@outlook.com>
"""

import tensorflow as tf
import numpy as np

from bnn_ops import binarize, hardtanh
from bnn_layers import BinaryDense, BinaryConv2D
from ebnn_ops import (
    binarize_np,
    binarize_np2real,
    np2float_c,
    np2int8_c,
    np_zip_bins2uint8_c,
)


class BinaryDenseBN(tf.keras.layers.Layer):
    """二值全连接层、批量归一化、softmax操作的复合层"""

    def __init__(
        self,
        units,
        use_bias=True,
        momentum=0.99,
        epsilon=0.001,
        is_binarized_activation=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bdense = BinaryDense(
            units=units,
            use_bias=use_bias,
            is_binarized_activation=is_binarized_activation,
        )
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

    def call(self, inputs, training=None):
        x = self.bdense(inputs)
        x = self.bn(x, training=training)
        return x

    def generate_c(self, layer_name):
        text = []
        # Binary Dense Layer
        binary_weights = binarize_np(self.bdense.kernel.numpy())
        text.append(np2int8_c(binary_weights, layer_name + "_w"))
        # 如果Binary Dense Layer定义了偏差
        if self.bdense.use_bias:
            text.append(np2float_c(self.bdense.bias.numpy(), layer_name + "_b"))
        else:
            text.append(f"float {layer_name}_b[{int(self.bdense.units)}] = {{0.0f}};")

        # Batch Normalization Layer
        # 缩放因子（gamma）、偏移量（beta）、移动平均的均值（moving_mean）和方差（moving_variance）
        # 通过方差计算标准差（standard deviation）
        text.append(np2float_c(self.bn.gamma.numpy(), layer_name + "_bn_gamma"))
        text.append(np2float_c(self.bn.beta.numpy(), layer_name + "_bn_beta"))
        text.append(np2float_c(self.bn.moving_mean.numpy(), layer_name + "_bn_mean"))
        text.append(np2float_c(self.bn.moving_variance.numpy(), layer_name + "_bn_var"))
        text.append(
            np2float_c(
                np.sqrt(self.bn.moving_variance.numpy() + self.bn.epsilon),
                layer_name + "_bn_std",
            )
        )

        text = "\n".join(text) + "\n"
        return text

    def generate_c_ori(self, layer_name):
        text = []
        # Binary Dense Layer
        weights = self.bdense.kernel.numpy()
        binary_weights = binarize_np2real(weights)
        # print("binary_weights.shape: ", binary_weights.shape)
        # binary_weights.shape:  (360, 10)
        binary_weights = binary_weights.T
        text.append(np_zip_bins2uint8_c(binary_weights, layer_name + "_bl_W", pad="1"))
        if self.bdense.use_bias:
            bias = self.bdense.bias.numpy()
            text.append(np2float_c(bias, layer_name + "_bl_b"))
        else:
            text.append(
                f"float {layer_name}_bl_b[{int(self.bdense.units)}] = {{0.0f}};"
            )
        # Batch Normalization Layer
        gamma = self.bn.gamma.numpy()
        beta = self.bn.beta.numpy()
        moving_mean = self.bn.moving_mean.numpy()
        moving_variance = self.bn.moving_variance.numpy()
        moving_std = np.sqrt(moving_variance + self.bn.epsilon)
        text.append(np2float_c(gamma, layer_name + "_bn_gamma"))
        text.append(np2float_c(beta, layer_name + "_bn_beta"))
        text.append(np2float_c(moving_mean, layer_name + "_bn_mean"))
        text.append(np2float_c(moving_std, layer_name + "_bn_std"))

        text = "\n".join(text) + "\n"
        return text


class BinaryConvPoolBNHT(tf.keras.layers.Layer):
    """FP32卷积层、池化层、批量归一化、HardTanh操作的复合层"""

    def __init__(
        self,
        filters,
        kernel_size,
        use_bias=False,
        strides=(1, 1),
        padding="valid",
        is_binarized_activation=True,
        pool_size=(2, 2),
        pool_strides=None,
        pool_padding="valid",
        momentum=0.99,
        epsilon=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bconv = BinaryConv2D(
            filters=filters,
            kernel_size=kernel_size,
            use_bias=use_bias,
            strides=strides,
            padding=padding,
            is_binarized_activation=is_binarized_activation,
        )
        self.pool = tf.keras.layers.MaxPooling2D(
            pool_size=pool_size, strides=pool_strides, padding=pool_padding
        )
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

    def call(self, inputs, training=None):
        x = self.bconv(inputs)
        x = self.pool(x)
        x = self.bn(x, training=training)
        x = binarize(x)
        return x

    def generate_c_ori(self, layer_name):
        text = []
        # Binary Conv2D Layer
        weights = self.bconv.kernel.numpy()
        # print("weights.shape: ", weights.shape)
        # weights.shape:  (3, 3, 1, 10)
        binary_weights = binarize_np2real(weights)
        binary_weights = binary_weights.reshape(binary_weights.shape[-1], -1)
        # print("binary_weights.shape: ", binary_weights.shape)
        # binary_weights.shape:  (10, 9)
        text.append(
            np_zip_bins2uint8_c(binary_weights, layer_name + "_bconv_W", pad="1")
        )

        if self.bconv.use_bias:
            bias = self.bconv.bias.numpy()
            text.append(np2float_c(bias, layer_name + "_bconv_b"))
        else:
            text.append(
                f"float {layer_name}_bconv_b[{int(self.bconv.filters)}] = {{0.0f}};"
            )

        # Batch Normalization Layer
        gamma = self.bn.gamma.numpy()
        beta = self.bn.beta.numpy()
        moving_mean = self.bn.moving_mean.numpy()
        moving_variance = self.bn.moving_variance.numpy()
        moving_std = np.sqrt(moving_variance + self.bn.epsilon)

        text.append(np2float_c(gamma, layer_name + "_bn_gamma"))
        text.append(np2float_c(beta, layer_name + "_bn_beta"))
        text.append(np2float_c(moving_mean, layer_name + "_bn_mean"))
        text.append(np2float_c(moving_std, layer_name + "_bn_std"))

        text = "\n".join(text) + "\n"
        return text
