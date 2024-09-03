"""ebnn

Note:
    File   : ebnn_numpy.py
    Author : Baiqi.Lu <ittuann@outlook.com>
"""

import numpy as np

from ebnn_ops import binarize_np


def fully_connected_inference(inputs, weights, bias=None):
    """
    Args:
        inputs: 输入数据, 形状为 (batch_size, input_dim)
        weights: 权重矩阵, 形状为 (input_dim, output_dim)
        bias: 偏置, 形状为 (output_dim,)
    Returns:
        输出数据, 形状为 (batch_size, output_dim)，每一行代表了一个样本在经过线性变换后的输出
    """
    output = np.dot(inputs, weights)
    if bias is not None:
        output += bias
    return output


def binarize_fully_connected_inference(inputs, weights, bias=None):
    """
    Args:
        inputs: 输入数据, 形状为 (batch_size, input_dim)
        weights: 权重矩阵, 形状为 (input_dim, output_dim)
        bias: 偏置, 形状为 (output_dim,)
    Returns:
        输出数据, 形状为 (batch_size, output_dim)，每一行代表了一个样本在经过线性变换后的输出
    """
    binary_weights = binarize_np(weights)
    output = np.dot(inputs, binary_weights)
    if bias is not None:
        # BNN隐藏层推理使用二值化的权重与二值化激活值计算，因此NumPy推导结果类型为np.int32
        # 需要显示转换为float64，以遵从NumPy的same_kind类型转换规则
        output = output.astype(np.float64)
        output += bias
    return output


def batch_normalization_inference(inputs, gamma, beta, mean, variance, epsilon=1e-3):
    """
    Args:
        inputs: 输入数据，形状为 (batch_size, dim)
        gamma: 缩放系数，形状为 (dim,)
        beta: 偏移量，形状为 (dim,)
        mean: 均值，形状为 (dim,)
        variance: 方差，形状为 (dim,)
        epsilon: 一个很小的数，防止除0
    Returns:
        输出数据，形状为 (batch_size, dim)
    """
    normalized = (inputs - mean) / np.sqrt(variance + epsilon)
    return gamma * normalized + beta


def softmax_inference(inputs):
    """
    Args:
        inputs: 输入数组, 形状为 (batch_size, classes)
    Returns:
        输出数据softmax概率值, 形状为 (batch_size, classes)
    """
    e_x = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def max_softmax_inference(inputs):
    """
    Args:
        inputs: 输入概率值数组, 形状为 (batch_size, classes)
    Returns:
        输出预测值, 形状为 (batch_size, )
    """
    return np.argmax(inputs, axis=1)


def sin_inference(inputs, w0=30):
    """
    Args:
        inputs: 输入数组, 形状为 (batch_size, dim)
    Returns:
        输出数据, 形状为 (batch_size, dim)
    """
    return np.sin(inputs * w0)
