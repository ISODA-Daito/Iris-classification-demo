import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class testmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.act = nn.ReLU()
        self.hidden = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 3)
        
    def forward(self, x):
        y = self.fc1(x)
        y = self.act(y)
        y = self.hidden(y)
        y = self.act(y)
        y = self.fc2(y)
        return y



#データをロード
iris = load_iris()
x_org, y_org = iris.data[:, 1:3], iris.target
x_train, x_test, y_train, y_test = train_test_split(
    x_org, y_org, test_size=0.33, random_state=42
)
x_train= torch.tensor(x_train).float()
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test)
label = iris.target_names
print(label)

# NNをインスタンス化
net = testmodel()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = net.parameters(), lr = 0.01, momentum = 0.01)


train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []


#train
for epoch in range(500):
    total_loss = 0
    net.train()
    correct = 0
    for feature, target in zip(x_train, y_train):
        optimizer.zero_grad()
        
        predict = net(feature)
        
        loss = loss_function(predict, target)
        
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()

        if target == torch.argmax(predict):
            correct += 1
    #validation
    net.eval()
    with torch.no_grad():
        val_total_loss = 0
        test_correct = 0
        for feature, target in zip(x_test, y_test):
            predict = net(feature)
            val_loss = loss_function(predict, target)
            val_total_loss += val_loss.item()
            if torch.argmax(predict) == target:
                test_correct += 1
    
    #result
    print(f"epoch {epoch}: train_loss = {total_loss / len(x_train)}, test_loss = {val_total_loss / len(y_test)}")
    print(f"accuracy, train: {correct / len(x_train) * 100:.1f}%; test: {test_correct / len(y_test) * 100:.1f}%")
    train_accuracy.append(correct / len(x_train) * 100)
    test_accuracy.append(test_correct / len(y_test) * 100)
    train_loss.append(total_loss)
    test_loss.append(val_total_loss)

# 結果をプロット
from matplotlib.colors import ListedColormap
markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(y_test.detach().numpy()))])

x_min ,x_max = x_test[:, 0].min() -1, x_test[:, 0].max() + 1
y_min, y_max = x_test[:, 1].min() -1, x_test[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x_min, x_max, step = 0.02),
                       np.arange(y_min, y_max, step = 0.02))
mesh1, mesh2 = np.meshgrid(np.arange(x_min, x_max, step = 0.1),
                           np.arange(y_min, y_max, step = 0.1))

test = np.vstack([xx1.ravel(), xx2.ravel()]).T
test2 = np.vstack([mesh1.ravel(), mesh2.ravel()]).T

pred_mesh = torch.argmax(net(torch.tensor(test).float()), dim = 1).detach().numpy()

m = nn.Softmax(dim = 1)
prob = m(net(torch.tensor(test2).float())).detach().numpy()*100


z = pred_mesh.reshape(xx1.shape)
zz = prob.reshape((*mesh1.shape, 3))


plt.figure()
plt.contourf(xx1, xx2, z, alpha = 0.3, cmap = cmap)

for i in range(3):
    sample = x_test[y_test == i]
    plt.scatter(sample[:, 0], sample[:, 1], c = colors[i], marker=markers[i], label = label[i])
plt.title("Iris Classification")
plt.xlabel("sepal width [cm]")
plt.ylabel("petal length [cm]")
plt.legend()


for k in range(3):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(mesh1, mesh2, zz[..., k], cmap = plt.cm.Grays, vmin = zz.min(), vmax = zz.max())
    for i in range(3):
        sample = x_test[y_test == i]
        ax.scatter(sample[:, 0], sample[:, 1], c = colors[i], marker=markers[i], label = label[i])
    plt.colorbar(mappable= heatmap, label = "max probability", ax=ax)
    ax.set_title(f"posterior probability distribution of {label[k]}")
    ax.set_xlabel("sepal width [cm]")
    ax.set_ylabel("petal length [cm]")
    ax.legend()

plt.figure()
epochs = [i+1 for i in range(len(train_loss))]
plt.plot(epochs, train_loss, label = "train loss")
plt.plot(epochs, test_loss, label = "test loss")
plt.legend()
plt.title("Loss")
plt.xlabel("epoch")
plt.ylabel("Average Loss")

plt.figure()
plt.plot(epochs, train_accuracy, label = "train")
plt.plot(epochs, test_accuracy, label = "test")
plt.legend()
plt.title("Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy [%]")
plt.show()
