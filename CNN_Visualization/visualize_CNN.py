
import torch
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from gradcam import GradCAM, GradCAMpp


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# load network
net = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11', pretrained=True)
net = net.eval().to(device)

# image to tensor transform
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

data_dir = './data/'
image_one_sample = datasets.ImageFolder(os.path.join(data_dir, 'visualize_samples'), transform)
img_t, class_names = next(iter(torch.utils.data.DataLoader(image_one_sample)))
image_raw = mpimg.imread(os.path.join(data_dir, 'visualize_samples')+'/Labrador/dog.jpg')

initial_class_pred = 208
x = img_t.to(device)
labels = torch.tensor([initial_class_pred])
labels = labels.to(device)

filename = "./imagenet_classes.txt"
all_labels = []

with open(filename, 'r', encoding="utf-8") as filehandle:
    for line in filehandle:
        all_labels.append(line)

##############################################
#  Assignment 1:
#  Visualise the input image and the transformed
#  (resampled & normalised) tensor image
##############################################

##################### Visualise the input image and the transformed image

plot_out = torchvision.utils.make_grid(img_t, nrow=10, normalize=True, padding=1)
image_transposed = plot_out.numpy().transpose((1, 2, 0))

plot_ = plt
fig = plot_.figure()
ax = fig.add_subplot(1, 2, 1)
imgplot = plot_.imshow(image_raw)
ax.set_title('Input image')
# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
ax = fig.add_subplot(1, 2, 2)
imgplot = plot_.imshow(image_transposed)
ax.set_title('Transformed input')
# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
plot_.show()


# ##################### print the top 10 classes

# we are not changing the network weights/biases in this lab
for param in net.parameters():
    param.requires_grad = False
print(net)

forward_outputs = net(x)

print_top_number = 10
_, predictedTop5_test = forward_outputs.topk(print_top_number)

probabilities = torch.sigmoid(forward_outputs.squeeze(0))
probabilities = probabilities.numpy() * 100.0
predictedTop5_test = predictedTop5_test.squeeze(0).numpy()

for x_i in range(print_top_number):
    print('id: ' + str(predictedTop5_test[int(x_i)]) + ' ' + str(all_labels[predictedTop5_test[int(x_i)]]) + ' ' + str(probabilities[int(x_i)]) + '%')






##############################################
#  Assignment 2:
#  To compute the l2 norms over the activation channels for each
#  of the 21 feature maps and to display them
##############################################

##################### Visualise the features map

fig1, axs1 = plt.subplots(nrows=6, ncols=4, figsize=(16, 20))
x_element1 = x
for (i, l) in enumerate(net.features):
    x_element1 = l.forward(x_element1)
    f = (x_element1.detach()**2).sum(dim=1).sqrt()[0]
    axs1.flat[i].imshow(f.cpu().numpy(), cmap='jet')
    axs1.flat[i].set_axis_off()
    axs1.flat[i].set_title("{}:{}".format(i, l.__class__.__name__))
fig1.show()

##################### Visualise the gradients map

# enable the changing of network weights/biases
for param in net.parameters():
    param.requires_grad = True

forward_outputs = net(x)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

_, preds = torch.max(forward_outputs, 1)
loss = criterion(forward_outputs, labels)

net.zero_grad()
optimizer.zero_grad()

loss.backward(retain_graph=True)


fig2, axs2 = plt.subplots(nrows=6, ncols=4, figsize=(16, 20))
x_element2 = x
for (i, l) in enumerate(net.features):
    xgcv2 = GradCAM(net, target_layer=l)
    mask = np.squeeze(xgcv2(x_element2)[0])
    axs2.flat[i].imshow(mask.cpu().numpy(), cmap='jet')
    axs2.flat[i].set_axis_off()
    axs2.flat[i].set_title("{}:{}".format(i, l.__class__.__name__))
fig2.show()

##################### Visualise the gradients map w.r.t raw image
import cv2
fig3, axs3 = plt.subplots(nrows=6, ncols=4, figsize=(16, 20))
x_element3 = x
for (i, l) in enumerate(net.features):
    xgcv2 = GradCAM(net, target_layer=l)
    mask = np.squeeze(xgcv2(x_element3)[0])

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    heatmap = heatmap[..., ::-1]  # gbr to rgb

    raw_img = abs(np.squeeze(x_element3)).numpy()
    raw_img = raw_img.transpose(1, 2, 0)

    cam = heatmap + np.float32(raw_img)  # overlap heatmap on the raw image
    cam = cam / np.max(cam)

    axs3.flat[i].imshow(np.uint8(255 * cam))
    # axs.flat[i].imshow(cam.cpu().numpy(), cmap='jet')
    axs3.flat[i].set_axis_off()
    axs3.flat[i].set_title("{}:{}".format(i, l.__class__.__name__))
fig3.show()


##############################################
#  Assignment 3:
#  To find input patterns that maximise the outputs of a given layer
#  (recall the work of Hubel and Wiesel). To find the input that
#  maximises the activation of an unit in that layer, we will numerically
#  optimize over a patch of the input image.
##############################################

def receptive_field(layer):
    rsize = 1
    rstride = 1
    for i in range(layer+1):
        l = net.features[i]
        if isinstance(l, torch.nn.Conv2d):
            rsize = rsize + (l.weight.size(2) - 1) * rstride
        if isinstance(l, torch.nn.MaxPool2d):
            rsize = rsize + rstride
            rstride *= 2
    return rsize

def activation_features(layer, x_element):
    for i in range(layer+1):
        l = net.features[i]
        x_element = l.forward(x_element)
    return x_element

def activation_max(layer_num, epsilon=None):

    S = receptive_field(layer_num)
    l = net.features[layer_num]
    channels = l.in_channels
    x_trick = torch.nn.Parameter(torch.zeros(channels, 3, S, S)).to(device)
    optimizer = torch.optim.Adam([x_trick], lr=0.001)

    apool = torch.nn.AvgPool2d(3, padding=0, stride=1)
    apad = torch.nn.ReplicationPad2d(1)

    for i_iteration in range(300):
        optimizer.zero_grad()
        f = activation_features(layer_num, x_trick)
        sz = f.size()
        object_f = -f[:, :, sz[2]//2, sz[3]//2].diag().sum()
        object_f.backward()
        optimizer.step()
        if epsilon is not None:
            with torch.no_grad():
              xx = apool(apad(x_trick))
              diff = x_trick - xx
              dn = torch.linalg.norm(diff.flatten(2), dim=2, ord=1.0) / (S * S)
              if dn.max() > epsilon:
                  x_trick.data[dn > epsilon] = xx[dn > epsilon]

    x_trick.data.clamp_(-1.0, 1.0)
    return x_trick

x_activation_max = activation_max(6, None)
grid = torchvision.utils.make_grid(x_activation_max, nrow=10, normalize=True, padding=1)
grid = grid.cpu().numpy().transpose(1, 2, 0)
fig4, axs4 = plt.subplots(nrows=1, ncols=1, figsize=(20, 13))
axs4.imshow(grid)
fig4.show()






