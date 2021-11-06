import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import shutil

# query if we have GPU
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', dev)

# transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#### Loading dataset ####
# datasets
dataset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

# divide data to train and test
batch_size_train = 128
batch_size_test = 1000
validation_split = .1
n_epochs = 3
learning_rate = 0.001
momentum = 0.9
log_interval = 42

# Creating data indices for training and validation splits:
validation_split = .1
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

shuffle_dataset = True
random_seed= 42
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_test, sampler=test_sampler)

train_data_size = len(train_sampler)
test_data_size = len(test_sampler)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

#### Plotting Samples ####
# import matplotlib.pyplot as plt
# fig = plt
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# fig.show()

# lets verify how the loader packs the data
(data, target) = next(iter(train_loader))
# probably get [batch_size x 1 x 28 x 28]
print('Input  type:', data.type())
print('Input  size:', data.size())
# probably get [batch_size]
print('Labels size:', target.size())
# see number of trainig data:
n_train_data = len(train_loader)
print('Train data size:', train_data_size)

#### Constructing Model ####
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(32*7*7, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = x.view(-1, 32*7*7)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# model = nn.Sequential(
#     # Convolution 1
#     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
#     # nn.BatchNorm2d(16),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#     # Convolution 2
#     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
#     # nn.BatchNorm2d(32),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#     # Fully connected 1
#     nn.Linear(32 * 7 * 7, 100),
#     nn.ReLU(),
#     nn.Linear(100, 10)
# )

model = net()
model.to(dev)
print(model)
# loss function
loss = nn.CrossEntropyLoss(reduction='none')

#### optimizer ####
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#### Denoting Testing ####
def test():
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
      for data, target in test_loader:
          data = data.to(dev)
          target = target.to(dev)
          output = model(data)
          # test_loss += F.nll_loss(output, target, size_average=False).item()
          test_loss = loss(output, target).mean()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
  # test_loss /= test_data_size
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, test_data_size,
    100. * correct / test_data_size))
  return test_loss, 100. * correct / test_data_size

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

#### Training ####
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
history_dict = []
train_loss, train_acc, val_loss, val_acc = [], [], [], []
# history_dict["epochs"] = epoch
# history_dict["train_loss"] = train_loss
# history_dict["train_acc"] = train_acc
# history_dict["val_loss"] = val_loss
# history_dict["val_acc"] = val_acc

def main():
    print("Hello World!")
    # will accumulate total loss over the dataset
    L = 0
    history_train_num = 0
    PATH = './results/'
    for epoch in range(n_epochs):
        # loop fetching a mini-batch of data at each iteration
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(dev)
            target = target.to(dev)
            # flatten the data size to [batch_size x 784]
            # data_vectors = data.flatten(start_dim=1)
            # apply the network
            y = model.forward(data)
            # calculate mini-batch losses
            l = loss(y, target)
            # accumulate the total loss as a regular float number (important to sop graph tracking)
            L += l.sum().item()
            # the gradient usually accumulates, need to clear explicitly
            optimizer.zero_grad()
            # compute the gradient from the mini-batch loss
            l.mean().backward()
            # make the optimization step
            optimizer.step()
            correct_y = 0
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), train_data_size,
                           100.0 * batch_idx * len(data) / train_data_size, l.mean().item()))

                train_losses.append(l.mean().item())
                train_counter.append(
                    (batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))

            pred_y = y.data.max(1, keepdim=True)[1]
            correct_y += pred_y.eq(target.data.view_as(pred_y)).sum()

            train_loss_tensor = l.mean()
            train_acc_tensor = 100. * correct_y / batch_size_train

            train_loss.append([history_train_num, train_loss_tensor.item()])
            train_acc.append([history_train_num, train_acc_tensor.item()])
            history_train_num += 1

        # val_loss[:, history_train_num] = [epoch, val_loss_tensor.item()]
        val_loss_tensor, val_acc_tensor = test()
        val_loss.append([epoch, val_loss_tensor.item()])
        val_acc.append([epoch, val_acc_tensor.item()])

        checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss_tensor.item(),
                    }

        checkpoint_path = os.path.join(PATH, ('checkpoint_' + 'epoch' + str(epoch+1) + '.checkpoint'))
        best_model_path = os.path.join(PATH, ('checkpoint_' + 'best' + '.checkpoint'))
        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # ## save the model if validation loss has decreased
        # if valid_loss <= valid_loss_min:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        #     # save checkpoint as best model
        #     save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        #     valid_loss_min = valid_loss


    # save the model and the optimizer
    torch.save(model.state_dict(), './results/model_cpu.pth')
    torch.save(optimizer.state_dict(), './results/optimizer_cpu.pth')

    # saving the history
    history_dict = {
        "train_loss": train_loss,
         "train_acc": train_acc,
         "val_loss": val_loss,
         "val_acc": val_acc}
    file_to_write = open("history.pickle", "wb")
    pickle.dump(history_dict, file_to_write)

    print(f'Epoch: {epoch} mean loss: {L / n_train_data}')

    #### Ploting the Result ####
    train_loss_plot = np.zeros([2, len(train_loss)])
    train_acc_plot = np.zeros([2, len(train_acc)])
    for i in range(len(train_loss)):
        train_loss_plot[:, i] = np.array(train_loss[i])
        train_acc_plot[:, i] = np.array(train_acc[i])

    val_loss_plot = np.zeros([2, len(val_loss)])
    val_acc_plot = np.zeros([2, len(val_acc)])
    for i in range(len(val_loss)):
        val_loss_plot[:, i] = np.array(val_loss[i])
        val_acc_plot[:, i] = np.array(val_acc[i])

    fig1 = plt
    fig1.plot(train_loss_plot[0,:], train_loss_plot[1,:], color='blue')
    fig1.scatter(val_loss_plot[0,:]*422, val_loss_plot[1,:], color='red')
    fig1.legend(['Train Loss', 'Test Loss'], loc='upper right')
    fig1.xlabel('number of training examples seen')
    fig1.ylabel('negative log likelihood loss')
    fig1.show()

    fig2 = plt
    fig2.plot(train_acc_plot[0,:], train_acc_plot[1,:], color='blue')
    fig2.scatter(val_acc_plot[0,:]*422, val_acc_plot[1,:], color='red')
    fig2.legend(['Train Accuracy Rate', 'Test Accuracy Rate'], loc='upper right')
    fig2.xlabel('number of training examples seen')
    fig2.ylabel('The Accuracy Rate')
    fig2.show()


if __name__ == "__main__":
    main()