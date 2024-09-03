import torch


class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 保存前向传播时的输入，以便在后向传播时使用
        ctx.save_for_backward(input)
        # 根据x的值二值化，x >= 0时输出1，否则输出-1
        y = torch.where(input >= 0, torch.ones_like(input), -torch.ones_like(input))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # 从上下文中恢复前向传播时保存的输入
        (input,) = ctx.saved_tensors
        # 应用梯度掩码，当|x| <= 1时，梯度通过; 否则，梯度为0
        grad_mask = (torch.abs(input) <= 1).float()
        grad_input = grad_output * grad_mask
        return grad_input


if __name__ == "__main__":
    x = torch.tensor([0.5, -1.5, 0.0, 2.0], requires_grad=True)
    # 应用二值化函数
    y = Binarize.apply(x)
    print(y)
    # 对y进行操作，例如求和。然后反向传播
    y.sum().backward()
    # 打印x的梯度
    print(x.grad)
