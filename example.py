import torch
from torchvision import datasets
from PIL import Image
import os
import shutil

save_dir = "mnist_sample"

# 如果旧文件夹存在，先清空
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

dataset = datasets.MNIST(root="./data", train=True, download=False)

# 导出前 12 张
for i in range(12):
    img, label = dataset[i]
    img.save(os.path.join(save_dir, f"pic_{i}_num_{label}.jpg"))

print(f"✅ 导出完成！保存到：{save_dir}")
print("一共 12 张，占用空间很小")
