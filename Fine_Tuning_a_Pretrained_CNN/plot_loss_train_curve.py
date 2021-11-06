
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open(r"history_train_allPara.pickle", "rb") as input_file:  # history_train_allPara  history_train_fc
    history_file = pickle.load(input_file)

def ewma(x, alpha):
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


train_loss = history_file["train_loss"]
train_acc = history_file["train_acc"]
val_loss = history_file["val_loss"]
val_acc = history_file["val_acc"]

# lr_list = [1, 0.1, 0.01, 0.001, 0.0001]
lr_list = [0.001]

flag_data = 0
for lr_i in lr_list:
    # print(lr_i)

    train_loss_plot = np.zeros_like(np.array(train_loss[flag_data]))
    train_acc_plot = np.zeros_like(np.array(train_acc[flag_data]))
    val_loss_plot = np.zeros_like(np.array(val_loss[flag_data]))
    val_acc_plot = np.zeros_like(np.array(val_acc[flag_data]))

    for cross_validation_i in range(5):

        train_loss_plot += np.array(train_loss[flag_data])
        train_acc_plot += np.array(train_acc[flag_data])
        val_loss_plot += np.array(val_loss[flag_data])
        val_acc_plot += np.array(val_acc[flag_data])
        flag_data += 1

    train_loss_plot /= 5.0
    train_acc_plot /= 5.0
    val_loss_plot /= 5.0
    val_acc_plot /= 5.0


    alpha = 0.1
    train_loss_plot_ewm = ewma(train_loss_plot, alpha)
    fig1 = plt
    fig1.plot(range(len(train_loss_plot_ewm)), train_loss_plot_ewm, color='black')
    temp_ = int(len(train_loss_plot) / len(val_loss_plot))
    fig1.scatter(range(0, len(val_loss_plot)*temp_, temp_), val_loss_plot, color='red')
    fig1.plot(range(len(train_loss_plot)), train_loss_plot, color='blue', alpha=0.3)
    fig1.legend(['Train Loss with EWM', 'Train Loss', 'Test Loss'], loc='upper right')
    fig1.xlabel('number of training examples seen')
    fig1.ylabel('negative log likelihood loss')
    fig1.title('Loss_curve_lr_'+str(lr_i))
    # for x0, y0 in zip(range(0, len(val_loss_plot)*temp_, temp_), val_loss_plot): plt.quiver(x0, y0 + 0.2, 0, -1, color='g', width=0.01)  # 绘制箭头
    # fig1.show()
    title = 'WholePara_Loss_curve_lr_'+str(lr_i)
    fig1.savefig('%s.jpg' % title, bbox_inches='tight')
    fig1.close()

    alpha = 0.1
    train_acc_plot_ewm = ewma(train_acc_plot, alpha)
    fig2 = plt
    fig2.plot(range(len(train_acc_plot_ewm)), train_acc_plot_ewm, color='black')
    temp_ = int(len(train_acc_plot)/len(val_acc_plot))
    fig2.scatter(range(0, len(val_acc_plot)*temp_, temp_), 100 * val_acc_plot, color='red')
    fig2.plot(range(len(train_acc_plot)), train_acc_plot, color='blue', alpha=0.3)
    fig2.legend(['Train Accuracy Rate with EWM', 'Train Accuracy Rate', 'Test Accuracy Rate'], loc='lower right')
    fig2.xlabel('number of training examples seen')
    fig2.ylabel('The Accuracy Rate')
    fig1.title('Accuracy_curve_lr_'+str(lr_i))
    # for x0, y0 in zip(range(0, len(val_acc_plot)*temp_, temp_), val_acc_plot): plt.quiver(x0, y0 + 10, 0, -1, color='g', width=0.01)  # 绘制箭头
    # fig2.show()
    title = 'WholePara_Accuracy_curve_lr_'+str(lr_i)
    fig2.savefig('%s.jpg' % title, bbox_inches='tight')
    fig2.close()

