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
baseImgPath = "2023-09-02_17-39-32_502.jpeg"
# baseImgPath = "scanning_images/2023-09-01_11-25-16_677.jpeg"
searchFolder = "scanning_images/"
outputFolder = "output_matches/"
model_image_size = 160 # assume square image

# load the base image with pillow
baseImg = Image.open(baseImgPath)
# baseImg = torchvision.io.read_image(baseImgPath).to(device)

#load personal model
#TODO: load model

# load pretrained_model
model = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

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
    if path.is_file() and path.suffix == ".jpeg":
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


