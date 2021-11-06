
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle

plt.ion()   # interactive mode

# epoch_loss_train, epoch_acc_train, epoch_loss_val, epoch_acc_val = [1, 2], [2, 3], [3, 4], [4, 5]
# history = ((epoch_loss_train), (epoch_acc_train), (epoch_loss_val), (epoch_acc_val))

######################################################################
# Part 1 (2p)
# Load Data
######################################################################
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

num_classes = 10
data_dir = './data/butterflies/'
image_datasets_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
image_datasets_val = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

select_datasets_train = torch.utils.data.SubsetRandomSampler(list(range(len(image_datasets_train))))
select_datasets_val = torch.utils.data.SubsetRandomSampler(list(range(len(image_datasets_val))))

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets_train, batch_size=4,
                                             shuffle=False, sampler=select_datasets_train),
    'val': torch.utils.data.DataLoader(image_datasets_val, batch_size=4,
                                             shuffle=False, sampler=select_datasets_val)}

dataset_sizes = {
    'train': len(image_datasets_train),
    'val': len(image_datasets_val)}

class_names = image_datasets_train.classes

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

######################################################################
# Visualize a few images
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


######################################################################
# Part 2 (4p)
# Train the whole parameters under transfer learning strategy
######################################################################

######################################################################
# Training the model
# ------------------
def train_model(model, criterion, optimizer, scheduler, title, lr_i, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_train, acc_train, epoch_loss_val, epoch_acc_val = [], [], [], []

    for epoch in range(num_epochs):
        print('Start Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size_train = len(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    correct_y = preds.eq(labels.data.view_as(preds)).sum()
                    train_loss_tensor = loss.item() # .mean()
                    train_acc_tensor = 100.0 * correct_y / batch_size_train

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        loss_train.append(train_loss_tensor)
                        acc_train.append(train_acc_tensor)
                    # else:
                        # val_loss.append(train_loss_tensor.item())
                        # val_acc.append(train_acc_tensor.item())
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = 100.0 * running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                # epoch_loss_train.append(epoch_loss)
                # epoch_acc_train.append(epoch_acc)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
            else:
                epoch_loss_val.append(epoch_loss)
                epoch_acc_val.append(epoch_acc)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # save the model and the optimizer
                torch.save(model.state_dict(), 'model_best_'+title+'_lr_'+str(lr_i)+'.pth')
        print('Epoch End {}'.format(epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    history = ((loss_train), (acc_train), (epoch_loss_val), (epoch_acc_val))
    return model, history


######################################################################
# Visualizing the model predictions
# Generic function to display predictions for a few images
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

######################################################################
# Finetuning the convnet
# Load a pretrained model and reset final fully connected layer.
#
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 10)

######################################################################
# 1. Load the vgg11 model
model_ft = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11', pretrained=True)
# model_ft = torch.hub.load('pytorch/vision:v0.9.0', 'squeezenet1_0', pretrained=True)
print(model_ft)
# input_image_size = torch.randn(4, 3, 224, 224)
# from torch.utils.tensorboard import SummaryWriter
# with SummaryWriter(comment='vgg11_show')as w:
#     # w.add_embedding(dummy_input, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None)
#     w.add_graph(model_ft, (input_image_size, ))



######################################################################
# 2. Freeze all parameters of the model, so that they will not be trained, by
for param in model_ft.parameters():
    param.requires_grad = False

######################################################################
# 1 and 2. delete the “classifier” part that maps “features” to scores of 1000 ImageNet classes.Consider using torch.nn.
#          BatchNorm1d (after linear layers) or torch.nn.Dropout (after activations) inside your classifier block.
num_ftrs = model_ft.classifier[-1].in_features
model_ft.classifier[-1] = nn.Sequential(nn.Linear(in_features=num_ftrs, out_features=num_classes),
                                        nn.Dropout(p=0.2))

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

# ######################################################################
# # 3. Train and evaluate
# title = 'train_fc'
# lr_list = [1, 0.1, 0.01, 0.001, 0.0001]
# train_loss, train_acc, val_loss, val_acc = [], [], [], []
# for lr_i in range(len(lr_list)):
#
#     # Observe that all parameters are being optimized
#     optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr_list[lr_i], momentum=0.9)
#
#     # Decay LR by a factor of 0.1 every 7 epochs
#     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
#
#     for cross_validation_i in range(5):
#
#         model_ft, history_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, title, lr_list[lr_i],
#                                num_epochs=6)
#
#         train_loss.append(history_ft[0])
#         train_acc.append(history_ft[1])
#         val_loss.append(history_ft[2])
#         val_acc.append(history_ft[3])
#
# # saving the history
# history_dict = {
#     "train_loss": train_loss,
#     "train_acc": train_acc,
#     "val_loss": val_loss,
#     "val_acc": val_acc}
# file_to_write = open("history_train_fc.pickle", "wb")
# pickle.dump(history_dict, file_to_write)
#
# # visualize_model(model_ft)
#
#
#
# ######################################################################
# # Part 3 (4p)
# # Train the whole parameters under transfer learning strategy
# ######################################################################
# # ConvNet as fixed feature extractor
# model_ft = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11', pretrained=True)
# # model_ft = torch.hub.load('pytorch/vision:v0.9.0', 'squeezenet1_0', pretrained=True)
#
#
#
#
# for param in model_ft.parameters():
#     param.requires_grad = True
#
# num_ftrs = model_ft.classifier[-1].in_features
# model_ft.classifier[-1] = nn.Sequential(nn.Linear(in_features=num_ftrs, out_features=num_classes),
#                                         nn.Dropout(p=0.2))
#
# model_ft = model_ft.to(device)
# criterion = nn.CrossEntropyLoss()
#
# ######################################################################
# # Train and evaluate
# # Observe that all parameters are being optimized
# title = 'train_allPara'
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
#
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
#
# model_ft, history_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, title, 0.0001,
#                        num_epochs=6)
#
# # saving the history
# history_dict = {
#     "train_loss": train_loss,
#     "train_acc": train_acc,
#     "val_loss": val_loss,
#     "val_acc": val_acc}
# file_to_write = open("history_train_allPara.pickle", "wb")
# pickle.dump(history_dict, file_to_write)
#
# # visualize_model(model_ft)
#
# ######################################################################
# # Further Learning
#
#
