import torch
from d2l.torch import Animator


def accuracy(y_hat, y): #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter): #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator: #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n          # 创建 n 个初始值为 0.0 的列表
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)   # 把所有累加器重置为 0
    def __getitem__(self, idx):
        return self.data[idx]   # 支持像列表一样用索引访问：acc[0]、acc[1]

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""

    # 1. 将模型设置为训练模式（非常重要！）
    if isinstance(net, torch.nn.Module):
        net.train()

    # 2. 准备累加器：分别累加 [总损失, 正确预测数, 总样本数]
    metric = Accumulator(3)

    # 3. 遍历整个训练集（一个 epoch）
    for X, y in train_iter:
        # 前向传播：得到预测概率
        y_hat = net(X)

        # 计算损失
        l = loss(y_hat, y)

        # ==================== 梯度更新 ====================
        if isinstance(updater, torch.optim.Optimizer):
            # 使用 PyTorch 内置的优化器（第3.7节简洁实现时用）
            updater.zero_grad()      # 清空梯度
            l.mean().backward()      # 反向传播
            updater.step()           # 更新参数
        else:
            # 使用我们自己写的从零开始优化器（第3.6节）
            l.sum().backward()       # 反向传播
            updater(X.shape[0])      # 手动更新参数（传入 batch_size）

        # ==================== 累加指标 ====================
        metric.add(float(l.sum()),          # 当前 batch 的总损失
                   accuracy(y_hat, y),      # 当前 batch 的正确预测数量
                   y.numel())               # 当前 batch 的样本总数

    # 4. 返回本 epoch 的平均损失 和 平均准确率
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型(定义见第3章)"""

    # 1. 创建动画绘制器（实时画图）
    animator = Animator(xlabel='epoch',
                        xlim=[1, num_epochs],
                        ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    # 2. 开始训练多个 epoch
    for epoch in range(num_epochs):

        # 3. 训练一个 epoch（核心！） 调用上述的函数train_epoch_ch3进行一个周期的训练
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater) #最后返回平均训练损失, 平均训练准确率这样一个元组
        # train_metrics = (平均训练损失, 平均训练准确率)

        # 4. 在测试集上评估准确率
        test_acc = evaluate_accuracy(net, test_iter)

        # 5. 把当前 epoch 的结果画到图上
        animator.add(epoch + 1, train_metrics + (test_acc,))    #拼接为这样的元组(train_loss, train_acc, test_acc)

    # 6. 训练结束后的简单检查（防止模型没学好）
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, f"训练损失太高了: {train_loss}"
    assert train_acc <= 1 and train_acc > 0.7, f"训练准确率不正常: {train_acc}"
    assert test_acc <= 1 and test_acc > 0.7, f"测试准确率不正常: {test_acc}"