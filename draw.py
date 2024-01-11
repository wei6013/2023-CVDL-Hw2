import numpy as np
import matplotlib.pyplot as plt

train_loss = np.loadtxt('./MNIST_accuracy_loss/train_loss.txt')
val_loss = np.loadtxt('./MNIST_accuracy_loss/val_loss.txt')
train_accuracy = np.loadtxt('./MNIST_accuracy_loss/train_accuracy.txt')
val_accuracy = np.loadtxt('./MNIST_accuracy_loss/val_accuracy.txt')

#Loss
plt.figure(figsize=(6, 5))
plt.subplot(2, 1, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

#準確
plt.subplot(2, 1, 2)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.legend()
plt.title('Accuracy')

plt.tight_layout()
plt.savefig('Accuracy_Loss.png', dpi=300)
plt.show()

