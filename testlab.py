import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        # get the pretrained VGG19 network
        self.vgg = vgg19(pretrained=True)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]
        # print(self.features_conv)
        # get the max pool of the features stem
        self.max_pool = self.vgg.features[36]
        # print(self.max_pool)
        #  nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((2, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# define a 1 image dataset
dataset = datasets.ImageFolder(root='./data/Elephant', transform=transform)

# define the dataloader to load that single image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=2)

# initialize the VGG model
vgg = VGG()

# set the evaluation mode
vgg.eval()

# get the image from the dataloader
img_origin, _ = next(iter(dataloader))

# print(img.size())  # torch.Size([1, 3, 224, 224])
# get the most likely prediction of the model

index = 1
for index in range(2):
    img = img_origin
    # print(pred.size())  # torch.Size([1, 1000])
    pred = vgg(img)
    with torch.no_grad():
        pred_u = pred.argmax(dim=1)
    # print(type(pred_u))
    # print(pred_u.dtype)
    # print(pred[0])
    # get the gradient of the output with respect to the parameters of the model
    pred[index, pred_u[index]].backward()

    # pull the gradients out of the model
    gradients = vgg.get_activations_gradient()
    # print(gradients.size())  # torch.Size([1, 512, 14, 14])
    # pool the gradients across the channels
    # print(gradients[0].unsqueeze(0).size())
    pooled_gradients = torch.mean(gradients[index].unsqueeze(0), dim=[0, 2, 3])
    # print(pooled_gradients.size())  # torch.Size([512])
    # get the activations of the last convolutional layer
    activations = vgg.get_activations(img).detach()
    # print(activations.size())
    # print(activations.size())  # torch.Size([1, 512, 14, 14])
    # weight the channels by corresponding gradients
    for i in range(512):
        activations[0, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    # print(torch.mean(activations, dim=1).size())  # torch.Size([1, 14, 14])
    heatmap = torch.mean(activations[index].unsqueeze(0), dim=1).squeeze()
    # print(heatmap.size())  # torch.Size([14, 14])
    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # # draw the heatmap
    # plt.matshow(heatmap)
    #
    # plt.show()
    names = ['ele.jpeg', 'shark.jpeg']
    img2 = cv2.imread('./data/Elephant/data/{}'.format(names[index]))
    # print(img2.shape)
    # print(heatmap.numpy().shape)
    # print(img2.shape)
    # cv2.imshow('image', heatmap)
    # cv2.waitKey(0)
    heatmap = cv2.resize(heatmap.numpy(), (img2.shape[1], img2.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # print(heatmap)
    superimposed_img = heatmap * 0.4 + img2
    cv2.imwrite('./data/map{:02d}.jpg'.format(index), superimposed_img)
    # cv2.imshow('image', np.uint8(superimposed_img/np.amax(superimposed_img)*255))
    # cv2.waitKey(0)

# # Code source: Brian McFee
# # License: ISC
# # sphinx_gallery_thumbnail_number = 6
# import numpy as np
# import matplotlib.pyplot as plt

# import librosa
# import librosa.display

# y, sr = librosa.load(librosa.ex('trumpet'))

# D = librosa.stft(y)  # STFT of y
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# # plt.figure()
# # librosa.display.specshow(S_db)
# # plt.colorbar()


# # fig, ax = plt.subplots()
# # img = librosa.display.specshow(S_db, ax=ax)
# # fig.colorbar(img, ax=ax)

# # fig, ax = plt.subplots()
# # img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
# # ax.set(title='Now with labeled axes!')
# # fig.colorbar(img, ax=ax, format="%+2.f dB")

# # plt.show()

# plt.figure()
# librosa.display.waveplot(y, sr=sr)
# # ax[0].set(title='Monophonic')
# # ax[0].label_outer()

# fig, ax = plt.subplots()
# img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
# ax.set(title='Using a logarithmic frequency axis')
# fig.colorbar(img, ax=ax, format="%+2.f dB")

# plt.show()
