#include <float.h>  // FLT_MAX
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>  // 计时使用gettimeofday()

#include "mlp_ebnn_data.h"
#include "mlp_ebnn_mnist_data.h"

/**
 * 二值化全连接层推理
 * @inputs 指向输入数据数组的指针。使用一维数组模拟二维矩阵
 *
 * 全部使用float32进行计算并输出
 */
void binary_fully_connected_inference_all_fp32(const float *inputs, int batch_size, int input_dim, int output_dim,
                                               const float *weights, const float *bias, float *output) {
    for (int i = 0; i < batch_size; i++) {
        // 遍历批次内的每一个样本
        for (int j = 0; j < output_dim; j++) {
            // 遍历输出的每一个维度
            float sum = 0.0f;
            for (int k = 0; k < input_dim; k++) {
                // 计算内积
                // 需要将二维索引转换为一维索引
                // output[i][j] += inputs[i][k] * weights[k][j];
                sum += inputs[i * input_dim + k] * weights[k * output_dim + j];
            }
            // 处理偏置项
            if (bias != NULL) {
                sum += bias[j];
            }
            // 保存输出
            output[i * output_dim + j] = sum;
        }
    }
}

/**
 * 二值化全连接层推理
 *
 * 全部使用int8进行计算。输出类型为float
 */
void binary_fully_connected_inference_uint8(const uint8_t *inputs, int batch_size, int input_dim, int output_dim,
                                               const int8_t *weights, const float *bias, float *output) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < output_dim; j++) {
            // 内积变量为整形
            int sum_int = 0;
            for (int k = 0; k < input_dim; k++) {
                sum_int += inputs[i * input_dim + k] * weights[k * output_dim + j];
            }
            // 处理浮点数偏置项
            if (bias != NULL) {
                output[i * output_dim + j] = sum_int + bias[j];  // 隐式类型转换
            } else {
                output[i * output_dim + j] = sum_int;
            }
        }
    }
}

/**
 * 快速计算平方根的倒数
 */
float fast_inverse_square_root(float number) {
    const float x2 = number * 0.5f;
    const float threehalfs = 1.5f;
    union {
        float f;
        uint32_t i;
    } conv = {.f = number};
    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f *= threehalfs - (x2 * conv.f * conv.f);
    return conv.f;
}

/**
 * 批量归一化层推理
 */
void batch_normalization_inference(const float *inputs, int batch_size,
                                   const float *gamma, const float *beta, const float *mean, const float *variance, float epsilon,
                                   int dim,
                                   float *outputs) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim; j++) {
            // 计算归一化值
            float normalized = (inputs[i * dim + j] - mean[j]) / sqrt(variance[j] + epsilon);
            // 计算输出
            outputs[i * dim + j] = gamma[j] * normalized + beta[j];
        }
    }
}

/**
 * 使用使用标准差std的批量归一化层推理
 *
 * 使用标准差std代替方差var，减少开方运算
 */
void batch_normalization_inference_std(const float *inputs, int batch_size,
                                       const float *gamma, const float *beta, const float *mean, const float *std,
                                       int dim,
                                       float *outputs) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < dim; j++) {
            // 计算归一化值
            float normalized = (inputs[i * dim + j] - mean[j]) / std[j];
            // 计算输出
            outputs[i * dim + j] = gamma[j] * normalized + beta[j];
        }
    }
}

/**
 * Softmax 层推理
 */
void softmax_inference(const float *inputs, int batch_size, int classes, float *outputs) {
    for (int i = 0; i < batch_size; i++) {
        // 找到每个样本的最大值
        float maxInput = -FLT_MAX;
        for (int j = 0; j < classes; j++) {
            if (inputs[i * classes + j] > maxInput) {
                maxInput = inputs[i * classes + j];
            }
        }
        // 计算指数值，并累加得到指数和
        float sum = 0.0f;
        for (int j = 0; j < classes; j++) {
            outputs[i * classes + j] = exp(inputs[i * classes + j] - maxInput);
            sum += outputs[i * classes + j];
        }
        // 归一化
        for (int j = 0; j < classes; j++) {
            outputs[i * classes + j] /= sum;
        }
    }
}

/**
 * 从Softmax层推理输出的结果中，输出最大概率的索引
 */
void max_softmax_inference(const float *inputs, int batch_size, int classes, int *outputs) {
    for (int i = 0; i < batch_size; i++) {
        float maxProb = -FLT_MAX;
        int maxIndex = 0;
        for (int j = 0; j < classes; j++) {
            if (inputs[i * classes + j] > maxProb) {
                maxProb = inputs[i * classes + j];
                maxIndex = j;
            }
        }
        outputs[i] = maxIndex;  // 输出每个样本的最大概率值的索引
    }
}

int main() {
    // 输入数据
    const int input_dim = 784;
    const int output_dim = 10;

    // 中间层输出
    float fc_output[output_dim];
    float bn_output[output_dim];
    float softmax_output[output_dim];
    int max_softmax_output[1];

    int correct_predictions = 0;
    printf("Model Predictions vs Actual Labels:\n");

    // 处理每个测试样本
    for (int i = 0; i < 1; i++)
    {
        // 全连接层推理
        binary_fully_connected_inference_uint8(&test_data[i * input_dim], 1, input_dim, output_dim, binarize_fc1_w, binarize_fc1_b, fc_output);

        // 批量归一化推理
        batch_normalization_inference_std(fc_output, 1, binarize_fc1_bn_gamma, binarize_fc1_bn_beta, binarize_fc1_bn_mean, binarize_fc1_bn_std, output_dim, bn_output);

        // Softmax 推理
        softmax_inference(bn_output, 1, output_dim, softmax_output);

        // 最大概率索引
        max_softmax_inference(softmax_output, 1, output_dim, max_softmax_output);

        // 打印隐藏式输出
        printf("Fully Connected Output:\n");
        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < output_dim; j++) {
                printf("%f ", fc_output[i * output_dim + j]);
            }
            printf("\n");
        }

        // 打印预测结果
        printf("Sample %2d: Predicted = %d, Actual = %d", i, max_softmax_output[0], test_labels[i]);
        if (max_softmax_output[0] == test_labels[i]) {
            printf(" (Correct)\n");
            correct_predictions++;
        } else {
            printf(" (Incorrect)\n");
        }
    }

    // 计算并输出准确率
    float accuracy = (float)correct_predictions / 20 * 100.0f;
    printf("\nCorrect: %d; Incorrect: %d; Total: %d\n", correct_predictions, 20 - correct_predictions, 20);
    printf("Accuracy: %.2f%%\n", accuracy);

    return 0;
}
