"""ebnn

Note:
    File   : ebnn_ops.py
    Author : Baiqi.Lu <ittuann@outlook.com>
"""

import numpy as np


def binarize_np(x: np.ndarray) -> np.ndarray:
    """二值化函数的NumPy实现

    当x中的元素大于等于0时，返回+1；否则返回-1

    Args:
        x (np.ndarray): NumPy数组

    Returns:
        np.ndarray: +1/-1二值的实数数组
    """
    return np.where(x >= 0, 1, -1)


def binarize_np2real(x: np.ndarray) -> np.ndarray:
    """将NumPy数组转换成1/0二值的实数数组

    当x中的元素大于等于0时，返回1；否则返回0

    Args:
        x (np.ndarray): NumPy数组

    Returns:
        np.ndarray: 1/0二值的实数数组
    """
    return np.where(x >= 0, 1, 0).astype(int)


def np_zip_bins2uint8_c(xs, name, *, pad="0"):
    """将二值的NumPy二维数组输出压缩成 uint8 数组保存的二进制 C 语言代码

    Args:
        xs (np.ndarray): 输入可以是二维数组矩阵(如权重)，形状为 (input_dim, output_dim)
                         输入可以是一维数组向量，形状为 (dim, )
        name (str): 数组的名称
        pad (str, optional): 当xi长度小于8时，在xi的右侧填充至8所使用的字符

    Example:
        xs = np.array([[1], [0], [1]])
        print(np2uint8_c(xs, 'myArray'))
        uint8_t myArray[3] = {128,0,128};
    """
    zip_uint8 = []
    # 逐行遍历NumPy二维数组xs
    for x in xs:
        # 将x中的每个二值元素转换成二进制字符串，然后将8个二进制字符串连接起来
        for i in range(0, len(x), 8):
            # 每8个元素作为一组进行处理
            xi = x[i : i + 8]
            xi = "".join(map(str, xi))
            # 保证每个xi都将被处理成一个完整的8位长度二进制数。
            # 如果xi的长度小于8，则使用pad参数指定的字符（默认是'0'）在xi的右侧填充至8
            xi = xi.ljust(8, pad)
            # 将二进制字符串xi转换成一个整数。int(xi, 2)中的2表明xi是二进制数。
            xi = int(xi, 2)
            zip_uint8.append(xi)

    c_str = f"uint8_t {name}[{len(zip_uint8)}] = {{{','.join(map(str, zip_uint8))}}};"
    return c_str


def np2float_c(xs, name):
    """将NumPy数组输出成浮点 C 语言代码

    Args:
        xs (np.ndarray): 输入可以是二维数组矩阵(如权重)，形状为 (input_dim, output_dim)
                         输入可以是一维数组向量，形状为 (dim, )
        name (str): 数组的名称
    """
    xs = xs.flatten()
    # 保留六位小数，并删除尾随的零和不必要的小数点
    str_buf = [f"{x:.6f}f".rstrip("0").rstrip(".") for x in xs]

    c_str = f"float {name}[{len(xs)}] = {{{','.join(str_buf)}}};"
    return c_str


def np2int8_c(xs, name):
    """将NumPy数组输出成 int8 C 语言代码

    Args:
        xs (np.ndarray): 输入可以是二维数组矩阵(如权重)，形状为 (input_dim, output_dim)
                         输入可以是一维数组向量，形状为 (dim, )
        name (str): 数组的名称
    """
    xs = xs.flatten()
    str_buf = [str(int(x)) for x in xs]

    c_str = f"int8_t {name}[{len(xs)}] = {{{','.join(str_buf)}}};"
    return c_str


def np2uint8_c(xs, name):
    """将NumPy数组输出成 uint8 C 语言代码"""
    xs = xs.flatten()
    str_buf = [str(int(x)) for x in xs]
    c_str = f"uint8_t {name}[{len(xs)}] = {{{','.join(str_buf)}}};"
    return c_str
