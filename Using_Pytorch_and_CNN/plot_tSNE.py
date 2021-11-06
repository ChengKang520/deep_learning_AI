
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import mnist


learning_rate = 0.001
momentum = 0.9

# query if we have GPU
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', dev)

# transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# x.cpu().numpy()

continued_network = mnist.net()
network_state_dict = torch.load('./results/model_cpu.pth')
continued_network.load_state_dict(network_state_dict)

continued_optimizer = optim.SGD(continued_network.parameters(), lr=learning_rate,momentum=momentum)
optimizer_state_dict = torch.load('./results/optimizer_cpu.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)


#### Loading dataset for testing ####
testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128*10, shuffle=False, num_workers=0) # len(testset.targets)

flag_length = 0
output = []
# output_item = np.zeros([79, 128*50])
# target_item = np.zeros([79, 128])

features = []
def hook(module, input, output):
    features.append(output.clone().detach())

handle = continued_network.fc1.register_forward_hook(hook)

for data, target in test_loader:
    data = data.to(dev)
    target = target.to(dev)
    output = continued_network.forward(data)
    target = target.detach().numpy()

    AA = features[0].size()
    features = features[0].detach().numpy()
    if AA != (128, 50):
        break
    # output_item[flag_length, :] = output.reshape(1, 128*50)
    # flag_length += 1
handle.remove()


############################################################
X = features
Y = target
target_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Fit and transform with a TSNE
from sklearn.manifold import TSNE

# Visualize the data
target_ids = range(len(Y))
# Project the data in 2D
X_2d_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)

# from sklearn.neighbors import KNeighborsClassifier
# neighs = KNeighborsClassifier(n_neighbors=3)
# neighs.fit(X, Y)
# neighs_tsne = KNeighborsClassifier(n_neighbors=3)
# neighs_tsne.fit(X_2d_tsne, Y)
# A1 = neighs.predict(X_last)
# A2 = neighs_tsne.predict(X_2d_tsne_last)


from matplotlib import pyplot as plt
fig1 = plt
fig1.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, target_names):
    fig1.scatter(X_2d_tsne[Y == i, 0], X_2d_tsne[Y == i, 1], c=c, label=label)
fig1.legend()
fig1.show()


############################################################
# Plot Confusion Matrix, ROC curve, and PR curve

############################################################
from sklearn.metrics import confusion_matrix

# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

pred_target = output.data.max(1, keepdim=True)[1]
pred_target = pred_target.detach().numpy()
cm = confusion_matrix(target, pred_target, labels=None, sample_weight=None)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = np.around(cm, decimals=2)
f = plt.figure(figsize=(15, 10), dpi=300)
ax = plt.subplot(111)
df = pd.DataFrame(cm,index=classes,columns=classes)
sns.heatmap(df, cmap='Blues',annot=True, fmt='g')
ax.xaxis.tick_top()
ax.set_ylabel('True', fontsize=15)
ax.set_xlabel('Pred', fontsize=15)
ax.tick_params(axis='y',labelsize=15, labelrotation=45) # y axis
ax.tick_params(axis='x',labelsize=15) # x axis
f.savefig('%s.jpg' % 'MNIST_cm1', bbox_inches='tight')

############################################################
from sklearn.metrics import f1_score
ma_f1 = f1_score(target, pred_target, average='macro')
mi_f1 = f1_score(target, pred_target, average='micro')
print(ma_f1, mi_f1)

############################################################
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

num_classes = 10
scores = torch.softmax(output, dim=1).detach().numpy()  # out = model(data)
binary_label = label_binarize(Y, classes=list(range(num_classes)))  # num_classes=10

fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(binary_label[:, i], scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(binary_label.ravel(), scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= num_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(8, 8))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

for i in range(10):
    plt.plot(fpr[i], tpr[i], lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.savefig('Multi-class ROC.jpg', bbox_inches='tight')
plt.show()

############################################################
