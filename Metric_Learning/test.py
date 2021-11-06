import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import seaborn as sns
import pandas as pd

from datasets import TripletSampler, load_datasets, load_test
from models import TripletResNet
from losses import TripletLoss, TripletAngularLoss
from params import args

def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

# features = []
# def hook(module, input, output):
#     # module: model.conv2
#     # input :in forward function  [#2]
#     # output:is  [#3 self.conv2(out)]
#     features.append(output.clone().detach())
#     # output is saved  in a list

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize(args.img_size),
                                    transforms.CenterCrop(args.img_size),
                                    transforms.ToTensor()])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = load_test(args.test_json, transform)
    test_loader = DataLoader(test_dataset, batch_size=112)

    model = TripletResNet(args.output_dim)
    model = model.to(device)
    # model.load_state_dict(torch.load('./weights/' + args.experiment_name +'_features'+ str(args.output_dim) + '_Ture.pth'))  #
    model.load_state_dict(torch.load('./weights/resnet18_features100.pth')) #_features256_True
    model.eval()
    # model = torchvision.models.__dict__['resnet50'](pretrained=True)
    # model = model.to(device)
    # model.eval()

    # AA = model.model[-1]
    # handle_forward = model.model.register_forward_hook(hook)
    # handle_backward = model.model.register_backward_hook(hook)
    for param in model.parameters():
        param.requires_grad = True
    pred_metric = []
    y_test = []
    for i, (anchors, positives, negatives, labels) in enumerate(test_loader):
        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)
        metric = model(anchors)
        # metric_positives = model(positives)
        # metric_negatives = model(negatives)

        pred_metric.append(metric.detach().cpu().numpy())
        y_test.append(labels.detach().numpy())

    pred_metric = np.concatenate(pred_metric, 0)
    y_test = np.concatenate(y_test, 0)

    ############################################################
    # Plot t-SNE
    ############################################################
    y_reduced = TSNE(n_components=2, random_state=0).fit_transform(pred_metric)
    plt_tSNE = plt
    plt_tSNE.scatter(y_reduced[y_test == 0, 0], y_reduced[y_test == 0, 1], color='blue')
    plt_tSNE.scatter(y_reduced[y_test == 1, 0], y_reduced[y_test == 1, 1], color='red')
    plt_tSNE.scatter(y_reduced[y_test == 2, 0], y_reduced[y_test == 2, 1], color='green')
    # plt_tSNE.scatter(y_reduced[y_test == 3, 0], y_reduced[y_test == 3, 1], color='black')

    plt_tSNE.legend(['CAP', 'Covid Infected', 'Healthy Control', 'SPT'], loc='upper left')
    plt_tSNE.savefig(args.experiment_name + '_tSNE.png')
    plt_tSNE.show()

    ############################################################
    # Plot Comparing Result
    ############################################################
    import cv2
    from gradcam import GradCAM, GradCAMpp

    n_queries = 4
    xp_idx0 = np.random.choice(np.where(labels == 0)[0])
    xp_idx1 = np.random.choice(np.where(labels == 1)[0])
    xp_idx2 = np.random.choice(np.where(labels == 2)[0])
    # xp_idx3 = np.random.choice(np.where(labels == 3)[0])

    # queries = [xp_idx0, xp_idx1, xp_idx2, xp_idx3]
    queries = [xp_idx0, xp_idx1, xp_idx2]
    queries_metric = metric[queries, :]


    euc_dist = torch.cdist(queries_metric, metric, p=2)
    cloeset_50_value, cloeset_50_pos = torch.topk(euc_dist, 28, dim=1, largest=False, sorted=True, out=None)

    ##################### Visualise the gradients map w.r.t raw image
    #
    # # enable the changing of network weights/biases
    # # criterion = TripletLoss()
    # model.train()
    # criterion = nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    #
    # plt_Compare = plt
    # plt_Compare.clf()
    # f, axarr = plt_Compare.subplots(n_queries, 1, gridspec_kw={'wspace': 0, 'hspace': 0})
    # for i in range(n_queries):
    #     ax = axarr[i]
    #     # img_t = torch.cat((anchors[queries[i], :, :, :].unsqueeze(0), anchors[cloeset_50_pos[i, :], :, :, :]))
    #     img_t = anchors[cloeset_50_pos[i, 0:4], :, :, :]
    #     metric_single = model(img_t)
    #     metric_positives_single = model(positives[0:4, :, :, :])
    #     metric_negatives_single = model(negatives[0:4, :, :, :])
    #     loss = criterion(metric_single, metric_positives_single, metric_negatives_single)
    #
    #     model.zero_grad()
    #     loss.backward(torch.ones_like(metric_single), retain_graph=True)
    #     optimizer.step()
    #     optimizer.zero_grad()
    #
    #     for name, param in model.model.named_parameters():
    #         if name == '7.1.bn2.weight':
    #             param
    #             break
    #
    #     # for (i, l) in enumerate(model.model):
    #     xgcv2 = GradCAM(model, target_layer=model.model[1])
    #     mask = np.squeeze(xgcv2(img_t)[0])
    #
    #     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    #     heatmap = np.float32(heatmap) / 255
    #
    #     heatmap = heatmap[..., ::-1]  # gbr to rgb
    #
    #     raw_img = abs(np.squeeze(img_t)).numpy()
    #     raw_img = raw_img.transpose(1, 2, 0)
    #
    #     cam = heatmap + np.float32(raw_img)  # overlap heatmap on the raw image
    #     img_t = cam / np.max(cam)
    #     img_t = np.uint8(255 * img_t)
    #
    #     # axs.flat[i].imshow(cam.cpu().numpy(), cmap='jet')
    #
    #     plot_out = torchvision.utils.make_grid(img_t, nrow=4, normalize=True, padding=1)
    #     image_transposed = plot_out.numpy().transpose((1, 2, 0))
    #     ax.imshow(image_transposed)
    #     ax.axis('off')
    # plt_Compare.savefig(args.experiment_name + '_Comparring.png')
    # f.show()
    #
    # # handle_forward.remove()
    # # handle_backward.remove()

    ############################################################
    # Calculating mAP
    ############################################################
    predict_label = torch.cat(([labels[cloeset_50_pos[0, 1:]], labels[cloeset_50_pos[1, 1:]], labels[cloeset_50_pos[2, 1:]], labels[cloeset_50_pos[3, 1:]]]), 0)
    labels = th_delete(labels, queries)
    torch.sum(predict_label == labels)
    accuracy_rate = torch.sum(predict_label == labels)/len(labels)

    ############################################################
    # Plot Confusion Matrix, ROC curve, and PR curve

    ############################################################
    from sklearn.metrics import confusion_matrix

    # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # classes = ['CAP', 'Covid', 'HC', 'SPT']
    classes = ['COVID', 'NORMAL', 'TB']

    cm = confusion_matrix(labels, predict_label, labels=None, sample_weight=None)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.around(cm, decimals=2)
    f = plt.figure(figsize=(15, 10), dpi=300)
    ax = plt.subplot(111)
    df = pd.DataFrame(cm, index=classes, columns=classes)
    sns.heatmap(df, cmap='Blues', annot=True, fmt='g')
    ax.xaxis.tick_top()
    ax.set_ylabel('True', fontsize=15)
    ax.set_xlabel('Pred', fontsize=15)
    ax.tick_params(axis='y', labelsize=15, labelrotation=45)  # y axis
    ax.tick_params(axis='x', labelsize=15)  # x axis
    f.savefig('%s.jpg' % 'Confusion_Matrix', bbox_inches='tight')
    f.show()
    ############################################################
    from sklearn.metrics import f1_score

    ma_f1 = f1_score(labels, predict_label, average='macro')
    mi_f1 = f1_score(labels, predict_label, average='micro')
    print(ma_f1, mi_f1)

    ############################################################
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    num_classes = 4
    # scores = torch.softmax(output, dim=1).detach().numpy()
    scores = label_binarize(predict_label, classes=list(range(num_classes))) # out = model(data)
    binary_label = label_binarize(labels, classes=list(range(num_classes)))  # num_classes=10

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

    for i in range(num_classes):
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

    predict_label = torch.reshape(predict_label, (4, 27))
    labels = torch.reshape(labels, (4, 27))

    TP0 = torch.sum(predict_label[0, :] == labels[0, :])/len(labels[0, :])
    FP0 = torch.sum(predict_label[0, :] != labels[0, :])/len(labels[0, :])
    FN0 = torch.sum(predict_label[0, :] != labels[0, :])/len(labels[0, :])

    TP1 = torch.sum(predict_label[1, :] == labels[1, :])/len(labels[1, :])
    FP1 = torch.sum(predict_label[1, :] != labels[1, :])/len(labels[1, :])
    FN1 = torch.sum(predict_label[1, :] != labels[1, :])/len(labels[1, :])

    TP2 = torch.sum(predict_label[2, :] == labels[1, :])/len(labels[2, :])
    FP2 = torch.sum(predict_label[2, :] != labels[1, :])/len(labels[2, :])
    FN2 = torch.sum(predict_label[2, :] != labels[1, :])/len(labels[2, :])


    accuracy_rate_0 = torch.sum(predict_label[0, :] == labels[0, :])/len(labels[0, :])
    accuracy_rate_1 = torch.sum(predict_label[1, :] == labels[1, :])/len(labels[1, :])
    accuracy_rate_2 = torch.sum(predict_label[2, :] == labels[2, :])/len(labels[2, :])
    # accuracy_rate_3 = torch.sum(predict_label[3, :] == labels[3, :])/len(labels[3, :])

    from sklearn import metrics
    aps = []
    for y_t, y_s in zip(labels, predict_label):
        ap = metrics.average_precision_score(y_t, y_s)
        aps.append(ap)
    np.mean(np.array(aps))

    ha = 1

