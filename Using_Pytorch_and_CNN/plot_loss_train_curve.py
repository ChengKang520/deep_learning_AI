
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle


# query if we have GPU
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', dev)

with open(r"history.pickle", "rb") as input_file:
    history_file = pickle.load(input_file)

train_loss = history_file["train_loss"]
train_acc = history_file["train_acc"]
val_loss = history_file["val_loss"]
val_acc = history_file["val_acc"]

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

val_loss_plot = np.insert(val_loss_plot, 0, train_loss_plot[:, 0], axis=1)
val_acc_plot = np.insert(val_acc_plot, 0, train_acc_plot[:, 0], axis=1)
val_loss_plot[0, 1:] += 1
val_acc_plot[0, 1:] += 1


def ewma(x, alpha):
    '''
    Returns the exponentially weighted moving average of x.

    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}

    Returns:
    --------
    ewma: numpy array
          the exponentially weighted moving average
    '''
    # Coerce x to an array
    x = np.array(x)
    n = x.size

    # Create an initial weight matrix of (1-alpha), and a matrix of powers
    # to raise the weights by
    w0 = np.ones(shape=(n,n)) * (1-alpha)
    p = np.vstack([np.arange(i,i-n,-1) for i in range(n)])

    # Create the weight matrix
    w = np.tril(w0**p,0)

    # Calculate the ewma
    return np.dot(w, x[::np.newaxis]) / w.sum(axis=1)

alpha = 0.1
train_loss_plot_ewm = ewma(train_loss_plot[1,:], alpha)
fig1 = plt
fig1.plot(train_loss_plot[0,:], train_loss_plot_ewm, color='black')
fig1.scatter(val_loss_plot[0,:]*422, val_loss_plot[1,:], color='red')
fig1.plot(train_loss_plot[0,:], train_loss_plot[1,:], color='blue', alpha=0.3)
fig1.legend(['Train Loss with EWM', 'Train Loss', 'Test Loss'], loc='upper right')
fig1.xlabel('number of training examples seen')
fig1.ylabel('negative log likelihood loss')
for x0, y0 in zip(val_loss_plot[0,:]*422, val_loss_plot[1,:]): plt.quiver(x0, y0 + 0.2, 0, -1, color='g', width=0.01)  # 绘制箭头
fig1.show()


alpha = 0.1
train_acc_plot_ewm = ewma(train_acc_plot[1,:], alpha)
fig2 = plt
fig2.plot(train_acc_plot[0,:], train_acc_plot_ewm, color='black')
fig2.scatter(val_acc_plot[0,:]*422, val_acc_plot[1,:], color='red')
fig2.plot(train_acc_plot[0,:], train_acc_plot[1,:], color='blue', alpha=0.3)
fig2.legend(['Train Accuracy Rate with EWM', 'Train Accuracy Rate', 'Test Accuracy Rate'], loc='upper left')
fig2.xlabel('number of training examples seen')
fig2.ylabel('The Accuracy Rate')
for x0, y0 in zip(val_acc_plot[0,:]*422, val_acc_plot[1,:]): plt.quiver(x0, y0 + 10, 0, -1, color='g', width=0.01)  # 绘制箭头
fig2.show()

