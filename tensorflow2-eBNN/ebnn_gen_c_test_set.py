"""eBNN

Note:
    File   : ebnn_gen_c_test_set.py
    Author : Baiqi.Lu <ittuann@outlook.com>
"""

import numpy as np
import tensorflow.keras.datasets.mnist as mnist
from ebnn_ops import np2float_c, np2uint8_c


def gen_mnist_test_data(
    batch_size: int = 20,
    is_random: bool = False,
    is_normalize: bool = False,
    is_uint8: bool = True,
) -> str:
    """输出MNIST测试集的C语言字符串

    从MNIST数据集加载测试图像和标签，并将它们转换为C语言的字符串表示。

    Args:
        batch_size (int): 加载的测试数据的批量大小。
        is_random (bool): 是否随机选择批量大小的样本。
        is_normalize (bool): 是否将图像数据归一化到[0, 1]之间。
        is_uint8 (bool): 是否将图像数据输出为uint8类型。

    Returns:
        str: 包含测试图像和标签的C语言字符串。
    """
    test_images, test_labels = mnist.load_data()[1]

    if is_random:
        # 随机选择batch_size个样本
        indices = np.random.choice(len(test_images), size=batch_size, replace=False)
        selected_images = test_images[indices]
        selected_labels = test_labels[indices]
    else:
        # 选择前batch_size个样本
        selected_images = test_images[:batch_size]
        selected_labels = test_labels[:batch_size]

    if is_normalize:
        # 将图像数据归一化到[0, 1]之间
        selected_images = selected_images.astype("float32") / 255.0

    # 转换数据和标签为C语言字符串
    if is_uint8 and not is_normalize:
        images_c_str = np2uint8_c(selected_images, "test_data")
    else:
        images_c_str = np2float_c(selected_images, "test_data")
    labels_c_str = np2uint8_c(selected_labels, "test_labels")

    # 将图像和标签字符串合并为一个字符串
    return images_c_str + "\n" + labels_c_str


if __name__ == "__main__":
    print(gen_mnist_test_data(batch_size=1, is_uint8=True))
