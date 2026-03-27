import os

import torchvision
from torch.utils import data
from torchvision import transforms


def load_data_fashion_mnist(batch_size, resize=None):  # @save
    """下载Fashion-MNIST数据集,然后将其加载到内存中"""

    # 第1步：准备图像变换（transforms）
    trans = [transforms.ToTensor()]                    # 把 PIL Image 转为 torch.Tensor，并归一化到 [0,1]

    if resize:                                         # 如果传了 resize 参数（例如 64）
        trans.insert(0, transforms.Resize(resize))     # 在最前面插入 Resize 操作

    trans = transforms.Compose(trans)                  # 把多个 transform 组合成一个

    # 第2步：下载并加载训练集
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data",           # 数据保存路径
        train=True,               # True 表示加载训练集
        transform=trans,          # 应用上面定义的变换
        download=True             # 如果本地没有就自动下载
    )

    # 第3步：下载并加载测试集
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data",
        train=False,              # False 表示加载测试集
        transform=trans,
        download=True
    )

    # 第4步：返回两个 DataLoader
    return (
        data.DataLoader(mnist_train,
                        batch_size=batch_size,
                        shuffle=True,                    # 训练集要打乱
                        num_workers=0),

        data.DataLoader(mnist_test,
                        batch_size=batch_size,
                        shuffle=False,                   # 测试集不需要打乱
                        num_workers=0)
    )

if __name__ == "__main__":
    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break