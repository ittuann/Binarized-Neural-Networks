"""BNN层

Note:
    File   : bnn_layers.py
    Date   : 2024/01/31
    Author : Baiqi.Lu <ittuann@outlook.com>
"""

import tensorflow as tf
from keras import initializers

from bnn_ops import binarize


class BinaryDense(tf.keras.layers.Dense):
    """实现二值化的全连接层

    权重和激活都被二值化
    """

    def __init__(
        self,
        units,
        use_bias=True,
        is_binarized_activation=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        momentum=0.99,
        epsilon=1e-3,
        **kwargs
    ):
        """构造函数"""
        super().__init__(
            units,
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            **kwargs
        )
        self.is_binarized_activation = is_binarized_activation
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

    def build(self, input_shape):
        """创建层的权重"""
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name="weights",
            initializer=self.kernel_initializer,
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name="bias",
                initializer=self.bias_initializer,
                constraint=self.bias_constraint,
                regularizer=self.bias_regularizer,
                trainable=True,
            )
        else:
            self.bias = None

    def call(self, inputs, training=None):
        """层的前向传播"""
        # 因为二值化函数使用STE来计算梯度，虽然使用bin_x和bin_w计算，但在更新参数时更新的是权重的全精度版本w
        # 在量化感知训练QAT中，所有权重和激活都是伪量化的。
        # 将权重二值化
        binary_kernel = binarize(self.kernel)
        # 将输入/激活二值化
        if self.is_binarized_activation:
            binary_inputs = binarize(inputs)
        else:
            # 用于模型的第一层，不对输入的激活值进行二值化操作，以直接处理原始的RGB图像数据
            binary_inputs = inputs
        # 计算预激活值
        pre_activation = tf.matmul(
            binary_inputs, binary_kernel
        )  # tf.keras.backend.dot()函数也可也进行矩阵乘法

        # 如果使用偏置，则将其添加到输出中
        if self.use_bias:
            pre_activation = tf.keras.backend.bias_add(pre_activation, self.bias)

        # 应用批量归一化
        outputs = self.bn(pre_activation, training=training)

        return outputs

    def get_config(self):
        """返回BinaryDense的配置字典，用于保存和加载模型"""
        config = super().get_config()
        config.update(
            {
                "is_binarized_activation": self.is_binarized_activation,
                "bn": self.bn.get_config(),
            }
        )
        return config

    def get_binarize_weights(self):
        """返回BNN推理时使用的二值化权重"""
        return tf.cast(binarize(self.kernel), tf.int8)


class BinaryConv2D(tf.keras.layers.Conv2D):
    """实现二值化的卷积层"""

    def __init__(
        self,
        filters,
        kernel_size,
        use_bias=False,
        is_binarized_activation=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        momentum=0.99,
        epsilon=1e-3,
        **kwargs
    ):
        super().__init__(
            filters,
            kernel_size,
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            **kwargs
        )
        self.is_binarized_activation = is_binarized_activation
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)

    def build(self, input_shape):
        kernel_shape = self.kernel_size + (input_shape[-1], self.filters)
        self.kernel = self.add_weight(
            shape=kernel_shape,
            name="weights",
            initializer=self.kernel_initializer,
            constraint=self.kernel_constraint,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                name="bias",
                initializer=self.bias_initializer,
                constraint=self.bias_constraint,
                regularizer=self.bias_regularizer,
                trainable=True,
            )
        else:
            self.bias = None

    def call(self, inputs, training=None):
        binary_kernel = binarize(self.kernel)
        if self.is_binarized_activation:
            binary_inputs = binarize(inputs)
        else:
            binary_inputs = inputs
        pre_activation = tf.keras.backend.conv2d(
            binary_inputs,
            binary_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
        )
        if self.use_bias:
            pre_activation = tf.keras.backend.bias_add(
                pre_activation,
                self.bias,
            )
        # 应用批量归一化
        outputs = self.bn(pre_activation, training=training)

        return outputs

    def get_config(self):
        """返回BinaryDense的配置字典，用于保存和加载模型"""
        config = super().get_config()
        config.update(
            {
                "is_binarized_activation": self.is_binarized_activation,
                "bn": self.bn.get_config(),
            }
        )
        return config
