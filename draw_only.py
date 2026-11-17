import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

train_loss_list = [0.1618, 0.0497, 0.0332, 0.0253, 0.0187]
train_acc_list  = [0.9528, 0.9848, 0.9897, 0.9920, 0.9939]
test_loss_list  = [0.0545, 0.0398, 0.0429, 0.0325, 0.0341]
test_acc_list   = [0.9798, 0.9868, 0.9849, 0.9891, 0.9899]
epochs = [1, 2, 3, 4, 5]
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs,train_loss_list, marker='o', label="train loss")
plt.plot(epochs,test_loss_list, marker='o', label="test loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(epochs,train_acc_list, marker='o', label="train acc")
plt.plot(epochs,test_acc_list, marker='o', label="test acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.savefig("mnist_curve.png", dpi=300, bbox_inches="tight")
plt.show()
print("图片保存成功！")
