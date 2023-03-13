import copy


def train_one_epoch(epoch, dev, train_dl, model, loss_func, opti, mu):
    if mu != 0:
        global_model = copy.deepcopy(model)
    # 设置迭代次数
    data_sum = 0
    for epoch in range(epoch):
        for data, label in train_dl:
            data, label = data.to(dev), label.to(dev)
            # 模型上传入数据
            preds = model(data)
            # 计算损失函数
            loss = loss_func(preds, label)
            data_sum += label.size(0)
            # 正则项
            if mu != 0:
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = loss + (mu / 2) * proximal_term
            # 反向传播
            loss.backward()
            # 计算梯度，并更新梯度
            opti.step()
            # 将梯度归零，初始化梯度
            opti.zero_grad()
    # 返回当前Client基于自己的数据训练得到的新的模型参数
    weights = copy.deepcopy(model.state_dict())
    for k, v in weights.items():
        weights[k] = weights[k].cpu().detach()
    return data_sum, weights
