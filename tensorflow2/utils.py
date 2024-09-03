"""小工具函数

Note:
    File   : utils.py
    Author : Baiqi.Lu <ittuann@outlook.com>
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil
import datetime


class SaveModelCallback(tf.keras.callbacks.Callback):
    """自动保存模型回调类"""

    def __init__(self, save_freq=100, save_path="./model", save_name="model_epoch"):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_name = save_name

        # 确保保存路径存在
        self.save_path.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            self.model.save(self.save_path / f"{self.save_name}_{epoch + 1}.h5")
            print(f"\n Model auto saved at epoch {epoch + 1}")


def plt_training_validation_metrics(
    history: dict,
    epochs: int,
    title: str = "MNIST Training and Validation Metrics",
) -> None:
    """绘制训练和验证的准确率和损失图像

    Args:
        history: dict, 训练过程中的历史指标字典
        epochs: int, 训练轮数
        title: str, 图像标题

    Example:
        utils.plt_training_validation_metrics(history, epochs, "MNIST Training and Validation Metrics")
        plt.show()
    """
    if history is None:
        raise ValueError("history is None")
    if epochs is None:
        raise ValueError("epochs is None")

    plt.figure(figsize=(12, 4))
    plt.suptitle(title)
    epoch_range = range(1, epochs + 1)
    plt.subplot(1, 2, 1)
    plt.title("Accuracy")
    plt.ylabel("Accuracy val", fontsize=14)
    plt.xlabel("Epoch")
    plt.xticks(epoch_range)  # 设置x轴刻度
    plt.plot(epoch_range, history["accuracy"], label="Training", linestyle="--")
    plt.plot(epoch_range, history["val_accuracy"], label="Validation")
    plt.scatter(epoch_range, history["accuracy"], marker="x")  # 显示数据点
    plt.scatter(epoch_range, history["val_accuracy"], marker="o")
    plt.legend()  # 添加图例

    plt.subplot(1, 2, 2)
    plt.title("Loss")
    plt.ylabel("Loss val", fontsize=14)
    plt.xlabel("Epoch")
    plt.xticks(epoch_range)
    plt.plot(epoch_range, history["loss"], label="Training", linestyle="--")
    plt.plot(epoch_range, history["val_loss"], label="Validation")
    plt.scatter(epoch_range, history["loss"], marker="x")
    plt.scatter(epoch_range, history["val_loss"], marker="o")
    plt.legend()


def plt_predict_image(
    i, predictions_array, true_label, class_names, img, *, cmap=None
) -> None:
    """绘制预测结果图像

    Args:
        i (int): 要绘制图像的索引。
        predictions_array (np.ndarray): 模型的预测结果数组。
        true_label (np.ndarray): 测试集真实标签数组。
        class_names (Sequence[str]): 类别名称的序列，可以是列表或元组。
        img (np.ndarray): 图像数据数组。
        cmap (): matplotlib颜色映射选项。

    Example:
        plt_predict_image(0, predictions, test_labels, class_names, test_images, cmap="binary")
        plt.show()
    """
    true_label, img = true_label[i].item(), img[i]
    predicted_label = np.argmax(predictions_array[i])
    plt.grid(False)  # 不显示网格
    plt.xticks([])  # 不显示x轴刻度
    plt.yticks([])
    plt.imshow(img, cmap=cmap)

    if predicted_label == true_label:
        color = "green"
    else:
        color = "red"

    plt.xlabel(
        f"{class_names[predicted_label]} {100 * np.max(predictions_array[i]):.2f}% ({class_names[true_label]})",
        color=color,
    )


def plt_predict_value_array(i, predictions_array, true_label, class_names) -> None:
    """绘制预测结果概率图像

    Args:
        i (int): 要绘制预测的索引。
        predictions_array (np.ndarray): 模型的预测结果数组。
        true_label (np.ndarray): 测试集真实标签数组。
        class_names (Sequence[str]): 类别名称的序列，可以是列表或元组。

    Example:
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        utils.plt_predict_image(0, predictions, test_labels, class_names, test_images)
        plt.subplot(1,2,2)
        utils.plt_predict_value_array(0, predictions, test_labels, class_names)
        plt.tight_layout() # 自动调整子图之间的间距
        plt.show()
    """
    true_label = true_label[i].item()
    predicted_label = np.argmax(predictions_array[i])
    plt.grid(False)
    plt.yticks([])
    plt.ylim([0, 1])  # 设置y轴范围
    # x轴显示名称
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    # x轴显示数字
    # plt.xticks(range(len(class_names)))

    bar_plot = plt.bar(range(len(class_names)), predictions_array[i], color="gray")

    bar_plot[predicted_label].set_color("red")
    bar_plot[true_label].set_color("blue")


def plt_training_metrics(
    history: dict,
    epochs: int,
    title: str = "Training Metrics",
) -> None:
    """绘制衡量图像重建质量的评估指标图像"""
    if history is None:
        raise ValueError("history is None")
    if epochs is None:
        raise ValueError("epochs is None")

    plt.figure(figsize=(12, 4))
    plt.suptitle(title)
    epoch_range = range(1, epochs + 1)
    plt.subplot(1, 3, 1)
    plt.title("MSE")
    plt.ylabel("MSE Loss", fontsize=14)
    plt.xlabel("Epoch")
    # plt.xticks(epoch_range)  # 设置x轴刻度
    plt.plot(epoch_range, history["loss"], label="Training")
    # plt.scatter(epoch_range, history["loss"], marker=".")  # 显示数据点
    plt.legend()  # 添加图例
    plt.subplot(1, 3, 2)
    plt.title("PSNR")
    plt.ylabel("PSNR Evaluate", fontsize=14)
    plt.xlabel("Epoch")
    plt.plot(epoch_range, history["psnr_metric"], label="Training")
    # plt.scatter(epoch_range, history["psnr_metric"], marker=".")
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.title("SSIM")
    plt.ylabel("SSIM Evaluate", fontsize=14)
    plt.xlabel("Epoch")
    plt.plot(epoch_range, history["ssim_metric"], label="Training")
    # plt.scatter(epoch_range, history["ssim_metric"], marker=".")
    plt.legend()
    plt.tight_layout()  # 自动调整子图之间的间距


def get_tensorboard_dir(path: Path, *, is_clear_out: bool = True) -> str:
    """返回TensorBoard日志目录"""
    if is_clear_out and path.exists() and path.is_dir():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    return str(path / datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
