from d2l import torch as d2l


def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签(定义见第3章)"""

    # 第1行：只取测试集中的第一个 batch（非常重要！）
    for X, y in test_iter:
        break
    # 解释：test_iter 是 DataLoader，循环一次就拿到一个 batch（32或256张图片），
    #      然后 break 立即退出，不再继续加载后面的数据（节省时间）

    # 第2行：把真实标签数字转成文字（T-shirt、Trouser 等）
    trues = d2l.get_fashion_mnist_labels(y)
    # y 的形状是 (batch_size,)，例如 [0, 2, 9, ...]
    # 转换后得到字符串列表，例如 ['T-shirt', 'Pullover', 'Ankle boot', ...]

    # 第3行：模型预测 + 转成文字标签
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))  #后面去到最大概率的下标然后根据前面的get_fashion_mnist_labels获得这个下标对应的标签就是预测图片的标签信息

    # net(X)       → 输出概率矩阵，形状 (batch_size, 10)
    # .argmax(axis=1) → 每行取概率最大的类别索引（预测类别）
    # 再用 get_fashion_mnist_labels 转成文字

    # 第4行：把真实标签和预测标签拼成标题
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    # 最终标题样式示例：
    # "T-shirt\nPullover"   ← 第一行是真实标签，第二行是预测标签

    # 第5行：显示图片
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)),   # 取前 n 张图片，并展平成 (n, 28, 28)
        1, n,                           # 1 行 n 列
        titles=titles[0:n]              # 只显示前 n 个标题
    )

