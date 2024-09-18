# Binarized Neural Network (BNN) Implementation

This repository contains the implementation of Binarized Neural Networks (BNNs), using TensorFlow 2 and PyTorch , with inference using Numpy and C.

Binarized Neural Networks are a type of deep learning model where weights and activations are constrained to binary values, typically -1 and +1.

This approach reduces the computational complexity and memory footprint of neural networks, making them highly suitable for deployment on edge devices and in resource-constrained environments.

# TensorFlow 2 Implementation:

Implemented BNN using TensorFlow 2Keras.

## Usage:

Take BNN MLP on MNIST as an example. Import the `BinaryDense` layer from `bnn_layers` and use it:

```python
from bnn_layers import BinaryDense

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        BinaryDense(128, use_bias=is_use_bias, epsilon=epsilon, momentum=momentum, is_binarized_activation=False),
        BinaryDense(64, use_bias=is_use_bias, epsilon=epsilon, momentum=momentum),
        BinaryDense(10, use_bias=is_use_bias, epsilon=epsilon, momentum=momentum),
    ]
)
```

Show the results of model's binarized weights:

```python
for i, layer in enumerate(model.layers):
    if isinstance(layer, BinaryDense):
        binarized_weights = layer.get_binarize_weights()
        print(f"Layer {i} Name: {layer.name}, Inference used binarized Weights:")
        print(binarized_weights, "\n")
```

```plaintext
Layer 1 Name: binary_dense, Inference used binarized Weights:
tf.Tensor(
[[-1  1  1 ... -1 -1  1]
 [-1 -1  1 ...  1 -1 -1]
 ...
 [ 1 -1 -1 ... -1  1  1]
 [ 1  1 -1 ... -1  1  1]], shape=(784, 128), dtype=int8)

Layer 2 Name: binary_dense_1, Inference used binarized Weights:
tf.Tensor(
[[-1  1 -1 ...  1 -1  1]
 [-1  1 -1 ...  1 -1 -1]
 ...
 [-1 -1 -1 ...  1  1 -1]
 [ 1 -1 -1 ...  1 -1 -1]], shape=(128, 64), dtype=int8)

 Layer 3 Name: binary_dense_2, Inference used binarized Weights:
tf.Tensor(
[[-1  1  1  1  1 -1 -1 -1 -1  1]
 [ 1 -1 -1  1  1 -1  1  1  1  1]
 ...
 [ 1  1 -1 -1 -1 -1 -1  1  1 -1]
 [ 1 -1 -1  1  1  1  1  1 -1 -1]], shape=(64, 10), dtype=int8)
```

Results and model Evaluation:

![BNN_MLP_MNIST_Metrics](tensorflow2/plts/BNN_MLP_MNIST_Metrics.png)

![BNN_MLP_MNIST_Confusion](tensorflow2/plts/BNN_MLP_MNIST_Confusion.png)

![BNN_MLP_MNIST_Result](tensorflow2/plts/BNN_MLP_MNIST_Result.png)

# PyTorch Implementation:

Implemented BNN using PyTorch.

Usage: Import the `BinarizeLinear` layer from `bnn_layers`.

# eBNN TensorFlow 2 Implementation:

Implemented Embedded Binarized Neural Networks (eBNN) using TensorFlow 2Keras.

# Inference in Numpy:

The content is included in eBNN TensorFlow 2 Implementation.

# Inference in C:

It can be just compile the C code directly and run it. Provides 2 examples: `one_batch_example`, `two_layers_example`.

# Paper

Original Paper:

```plaintext
Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., & Bengio, Y. (2016). Binarized neural networks. Advances in neural information processing systems, 29.

Courbariaux, M., Bengio, Y., & David, J. P. (2015). Binaryconnect: Training deep neural networks with binary weights during propagations. Advances in neural information processing systems, 28.

McDanel, B., Teerapittayanon, S., & Kung, H. T. (2017). Embedded binarized neural networks. arXiv preprint arXiv:1709.02260.
```

Recommended reading & future works:

```plaintext
Nielsen, J., & Schneider-Kamp, P. (2024, July). BitNet B1. 58 Reloaded: State-of-the-Art Performance Also on Smaller Networks. In International Conference on Deep Learning Theory and Applications (pp. 301-315). Cham: Springer Nature Switzerland.

Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., ... & Wei, F. (2023). Bitnet: Scaling 1-bit transformers for large language models. arXiv preprint arXiv:2310.11453.

Kim, M., & Smaragdis, P. (2016). Bitwise neural networks. arXiv preprint arXiv:1601.06071.
```
