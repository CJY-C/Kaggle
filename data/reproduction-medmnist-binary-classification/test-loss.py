import torch
import torch.nn as nn

# 测试交叉熵损失函数

# 1. 生成数据
data = torch.randn(1, 3)
data = torch.Tensor(
    [[10, 0, 0]]
)

# 1.2 生成标签
label = torch.tensor([0])

# 1.2 查看数据

print(data)

# 1.3 查看标签
print(label)

# 2. 生成模型
cri = nn.CrossEntropyLoss()

# 3. 计算损失
loss = cri(data, label)

# 4. 查看损失
print(loss)