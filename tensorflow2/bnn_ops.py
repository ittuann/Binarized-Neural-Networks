"""BNN二值化操作

Note:
    File   : bnn_ops.py
    Date   : 2024/01/31
    Author : Baiqi.Lu <ittuann@outlook.com>
"""

import tensorflow as tf


@tf.custom_gradient
def binarize(x):
    """二值化函数（binarize）"""

    def grad(dy):
        """实现二值化函数梯度直通估计器STE（Straight Through Estimator）的效果

        通过使用STE可以让梯度通过，使得可以对二值化的参数进行有效的学习。
        因为sign函数的真实梯度在x!=0时是0，x=0时是无穷大。在反向传播时，将损失乘以0会导致损失梯度丢失。

        在后向传播(backward propagation)求梯度时, STE的规则如下:
            dy = x;
        并应用指示器函数（阈值化）
            当 |x| > 1,     dy = 0;
            当 |x| <= 1,    dy = x;

        在后向传递中，grad()会收到一个张量dy，它表示相对于函数输出的损失梯度。
        将dy作为相对于输入的梯度，直接传递梯度，让梯度直接流过内部输出。
        同时应用指示器函数（阈值化）。当一个输入梯度到来时，经过阈值处理后就直接输出。
        这保留了梯度的信息，并在x值过大时取消梯度。
        当x值过大时不取消梯度会恶化性能。阈值化能防止具有较大激活值（|x| > 1）的神经元被更新，起到一个门的作用。

        类似于 Hard Tanh 硬双曲正切函数
        """

        # 计算梯度掩模（指示器）
        grad_mask = tf.cast(tf.abs(x) <= 1, x.dtype)
        return dy * grad_mask

    # 在前向传播(forward propagation)时, 输出如下:
    #     当 x <   0.0, y = -1
    #     当 x >=  0.0, y = 1
    # sign(x)是当x=0时返回0
    # 返回一个元组：实际的二值化结果和梯度函数

    # y = tf.sign(x)
    y = tf.where(x >= 0, tf.ones_like(x), -tf.ones_like(x))
    return y, grad


def hardtanh(x):
    """硬双曲正切函数（Hard Tanh）实现

    x < -1,         HT(x) = -1
    -1 <= x <= 1,   HT(x) = x
    x > 1,          HT(x) = 1
    """
    return tf.maximum(-1.0, tf.minimum(1.0, x))


def test_binarize():
    """验证二值化binarize函数，并测试在进行反向传播时是否能正确地按STE预期传递梯度"""
    # 一些假数据和标签用于测试
    fp32_values = tf.Variable([-1.5, -0.5, 0.1, 0.5, 0], dtype=tf.float32)
    y_true = tf.constant([1.0, -1.0, 1.0, -1.0, 1], dtype=tf.float32)

    # 使用GradientTape来记录前向传播运算过程，以便自动计算梯度
    with tf.GradientTape() as tape:
        # 向前传播。应用二值化函数
        bin_values = binarize(fp32_values)
        # 返回每个元素的误差平方
        errors = tf.square(bin_values - y_true)
        # 均方误差损失
        loss = tf.reduce_mean(errors)

    # 计算梯度
    grads = tape.gradient(loss, fp32_values)

    print(f"Test float32 value:     {fp32_values.numpy()}")
    print(f"Binary value (Forward): {bin_values}")
    print(f"True value:             {y_true.numpy()}")
    print(f"Error (binarize):       {errors.numpy()}")
    print(f"Loss:                   {loss.numpy()}")
    print(f"Gradient (Backward):    {grads.numpy()}")


if __name__ == "__main__":
    test_binarize()
