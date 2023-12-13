import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import itertools
from matplotlib import image, pyplot as plt
import glob as glob
from PIL import Image
from pathlib import Path

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from facenet_pytorch import MTCNN, InceptionResnetV1

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# Detect if we have a GPU available
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
# if torch.backends.mps.is_available():
#   print("Using the GPU!")
# else:
#   print("WARNING: Could not find GPU! Using CPU only. If you want to enable GPU, please to go Edit > Notebook Settings > Hardware Accelerator and select GPU.")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# using facenet_pytorch for pretrained baseline + mtcnn face extraction
# constants
baseImgPath = "George_W_Bush_0001.jpg"
# baseImgPath = "scanning_images/2023-09-01_11-25-16_677.jpeg"
searchFolder = "scanning_images/"
outputFolder = "output_matches/"
model_image_size = 112 # assume square image
pretrained_treshold = 0.9
selftrained_threshold = 0.9

# load the base image with pillow
baseImg = Image.open(baseImgPath)
# baseImg = torchvision.io.read_image(baseImgPath).to(device)

#load personal model
class Facenet_NN1(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # torch.nn.LocalResponseNorm(size, alpha=0.0001, beta=0.75, k=1.0)
        self.conv1 = nn.Conv2d(3, 64, 7,padding=3, stride=2)
        self.pool1 = nn.MaxPool2d(3, padding=1, stride=2)
        self.rnorm1 = nn.LocalResponseNorm(64)

        self.conv2a = nn.Conv2d(64, 64, 1, stride=1)
        self.conv2 = nn.Conv2d(64, 192, 3,padding=1, stride=1)
        self.rnorm2 = nn.LocalResponseNorm(192)
        self.pool2 = nn.MaxPool2d(3, padding=1, stride=2)

        self.conv3a = nn.Conv2d(192, 192, 1, stride=1)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1, stride=1)
        self.pool3 = nn.MaxPool2d(3, padding=1, stride=2)

        self.conv4a = nn.Conv2d(384, 384, 1, stride=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1, stride=1)

        self.conv5a = nn.Conv2d(256, 256, 1, stride=1)
        self.conv5 = nn.Conv2d(256, 256, 3,padding=1, stride=1)

        self.conv6a = nn.Conv2d(256, 256, 1, stride=1)
        self.conv6 = nn.Conv2d(256, 256, 3,padding=1, stride=1)
        self.pool4 = nn.MaxPool2d(3,padding=1, stride=2)

        #todo what is concat layer?
        self.fc1 = nn.Linear(256*4*4, 128*4*4)
        # self.maxout1 = nn.AdaptiveMaxPool2d((32,1))
        self.fc2 = nn.Linear(128 * 4 *4, 128 * 4)
        # self.maxout2 = nn.MaxPool2d(2)

        self.fc7128 = nn.Linear(128*4,128)




    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.rnorm1(self.pool1(F.relu(self.conv1(x))))
        x2 = self.pool2(self.rnorm2(F.relu(self.conv2(self.conv2a(x1)))))
        x3 = self.pool3(F.relu(self.conv3(self.conv3a(x2))))
        x4 = F.relu(self.conv4(self.conv4a(x3)))
        x5 = F.relu(self.conv5(self.conv5a(x4)))
        x6 = self.pool4(F.relu(self.conv6(self.conv6a(x5))))
        x6i = torch.flatten(x6, 1)
        x7 = self.fc1(x6i)
        # x7i = torch.unflatten(x7, 1, (128, 7, 7))
        # x8 = self.maxout1(x7i)
        # x8i = torch.flatten(x8, 1)
        x9 = self.fc2(x7)
        # x9i = torch.unflatten(x9, 1, (128, 32))
        # x10 = self.maxout2(x9i)
        # x10i = torch.flatten(x10, 1)
        x11 = self.fc7128(x9)
        # normalize the output to a unit vector
        x11 = F.normalize(x11)

        #x7 = self.maxout2(self.fc2(self.maxout1(self.fc1(x6))))
        # ─Conv2d: 1-16                           [1, 256, 6, 6]            590,080
        # ─MaxPool2d: 1-17                        [1, 256, 2, 2]            --
        # return self.fc1(x6)
        return x11
        # return self.fc7128(x7)
        # return x5

model = Facenet_NN1().to(device)
model.load_state_dict(torch.load("model_final.pth", map_location=torch.device('cpu')))
model.eval()
# load pretrained_model
# model = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

# load mtcnn 
mtcnn = MTCNN(
    image_size=model_image_size, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device,
    keep_all=True
)

# detect face in base image
# print(baseImg.shape)
x_aligned, prob = mtcnn(baseImg, return_prob=True)
print(prob)
print(x_aligned.shape)
# print face
# plt.imshow(x_aligned.permute(1, 2, 0).int().numpy())
# save image as base_face.jpg
# torchvision.utils.save_image(x_aligned[0], "base_face0.jpg")
# torchvision.utils.save_image(x_aligned[1], "base_face1.jpg")
# torchvision.utils.save_image(x_aligned[2], "base_face2.jpg")
# torchvision.utils.save_image(x_aligned[3], "base_face3.jpg")

baseEmbedding = model(x_aligned[0].unsqueeze(0))[0]
print(baseEmbedding.shape)
# print(baseEmbedding)


# iterate through all images in searchFolder with pathlib
for path in Path(searchFolder).iterdir():
    if path.is_file() and (path.suffix == ".jpg" or path.suffix == ".jpeg"):
        print(path)
        # load image
        img = Image.open(path)
        # detect face
        faces, probs = mtcnn(img, return_prob=True)
        print(probs)
        # print(x_aligned)
        # print face
        # plt.imshow(x_aligned.permute(1, 2, 0).int().numpy())
        # save image as base_face.jpg
        # torchvision.utils.save_image(x_aligned, "search_face.jpg")
        # get embedding
        # print(faces.shape)
        print(type(faces))
        # print(probs.shape)
        if faces is not None:
            searchEmbeddings = model(faces)
            for i in range(len(probs)):
                if probs[i] > 0.9:
                    embed = searchEmbeddings[i]
                    print(embed.shape)
                    print(type(embed))

                    # compare embeddings
                    dist = torch.linalg.norm(baseEmbedding - embed, ord=2)
                    print(dist)
                    if dist < selftrained_threshold: # manual tuning
                        print("match!")
                        # save image to output folder
                        torchvision.utils.save_image(faces[i], outputFolder + path.name)
                        # save full image to output folder
                        img.save(outputFolder + path.stem + "_original.jpeg")
                        break
                        # break


