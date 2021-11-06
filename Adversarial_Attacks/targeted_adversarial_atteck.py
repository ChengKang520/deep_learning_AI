
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, models, transforms
from torch.autograd.gradcheck import zero_gradients
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

#  refer to: https://savan77.github.io/blog/imagenet_adv_examples.html

##############################################
#  Assignment 4:
#  To implement a targeted iterative adversarial attack.
##############################################

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

x = img_t.to(device)
filename = "./imagenet_classes.txt"
all_labels = []
with open(filename, 'r', encoding="utf-8") as filehandle:
    for line in filehandle:
        all_labels.append(line)

initial_class_pred = 208  #  Labrador retriever
labels = torch.tensor([initial_class_pred])
labels = labels.to(device)

target_class_id = 892   #  wall clock -> adversarial target
labels_adversarial_attack = torch.tensor([target_class_id])
labels_adversarial_attack = labels_adversarial_attack.to(device)


epsilon = 0.05
# epsilons = [0, .05, .1, .15, .2, .25, .3]
# # FGSM attack code
# def fgsm_attack(image, epsilon, data_grad):
#     # Collect the element-wise sign of the data gradient
#     sign_data_grad = data_grad.sign()
#     # Create the perturbed image by adjusting each pixel of the input image
#     perturbed_image = image + epsilon*sign_data_grad
#     # Adding clipping to maintain [0,1] range
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     # Return the perturbed image
#     return perturbed_image


def visualize(x, x_adv, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob):

    x = x.squeeze(0).detach().numpy() # remove batch dimension # B X C H X W ==> C X H X W
    x = np.transpose(x, (1, 2, 0))  # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)

    x_adv = x_adv.squeeze(0).detach().numpy()
    x_adv = np.transpose(x_adv, (1, 2, 0))  # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)

    x_grad = x_grad.squeeze(0).detach().numpy()
    x_grad = np.transpose(x_grad, (1, 2, 0))
    x_grad = np.clip(x_grad, 0, 1)

    figure, ax = plt.subplots(1, 3, figsize=(18, 8))
    ax[0].imshow(x)
    ax[0].set_title('Clean Example', fontsize=20)

    ax[1].imshow(x_grad)
    ax[1].set_title('Perturbation', fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].imshow(x_adv)
    ax[2].set_title('Adversarial Example', fontsize=20)

    ax[0].axis('off')
    ax[2].axis('off')

    ax[0].text(1.1, 0.5, "+{}*".format(round(epsilon, 3)), size=15, ha="center",
               transform=ax[0].transAxes)

    ax[0].text(0.5, -0.13, "Prediction: {}\n Probability: {}".format(clean_pred, clean_prob), size=15, ha="center",
               transform=ax[0].transAxes)

    ax[1].text(1.1, 0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

    ax[2].text(0.5, -0.13, "Prediction: {}\n Probability: {}".format(adv_pred, adv_prob), size=15, ha="center",
               transform=ax[2].transAxes)

    plt.show()



output = net.forward(x)
output_probs = F.softmax(output, dim=1)
output_probs = output_probs.detach().numpy()
x_pred = np.argmax(output_probs)
x_pred_prob = output_probs[0, x_pred]

# Set requires_grad attribute of tensor. Important for Attack
x.requires_grad = True
# Zero all existing gradients
net.zero_grad()

epsilon = 0.25
num_steps = 5
alpha = 0.025
img_variable = x  #in previous method we assigned it to the adversarial img
for i in range(num_steps):
    zero_gradients(img_variable)
    output = net.forward(img_variable)
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(output, labels_adversarial_attack)
    loss_cal.backward()
    x_grad = alpha * torch.sign(img_variable.grad.data)
    adv_temp = img_variable.data - x_grad
    total_grad = adv_temp - x
    total_grad = torch.clamp(total_grad, -epsilon, epsilon)
    x_adv = x + total_grad
    img_variable.data = x_adv

output_adv = net.forward(img_variable)
output_adv_probs = F.softmax(output_adv, dim=1).detach().numpy()
x_adv_pred = np.argmax(output_adv_probs)
x_adv_pred_prob = output_adv_probs[0, x_adv_pred]


x_pred = all_labels[x_pred]
x_adv_pred = all_labels[x_adv_pred]

visualize(x, img_variable.data, total_grad, epsilon, x_pred, x_adv_pred, x_pred_prob,  x_adv_pred_prob)


