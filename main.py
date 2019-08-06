import torch
import torch.nn as nn

from torch.autograd import Variable

import matplotlib.pyplot as plt

from torchvision import datasets, models, transforms


from model import Casae,ConvAutoencoder

import numpy as np

import os

if __name__== "__main__":

    #Data preparation

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=1),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),

            #transforms.Normalize([0.54431105, 0.5803863, 0.53637147], [0.00527598, 0.00665597, 0.00575981])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            #   transforms.CenterCrop(224),
            transforms.ToTensor()
            #transforms.Normalize([0.54431105, 0.5803863, 0.53637147], [0.00527598, 0.00665597, 0.00575981])
        ]),
    }

    data_dir="/home/tondelli/Desktop/DatasetPalline/dataset1/"

    dset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train']}


    dset_loaders = {x: torch.utils.data.DataLoader(dset[x], batch_size=1,
                                                   shuffle=True, num_workers=4)
                    for x in ['train']}

    criterion= torch.nn.MSELoss()








    model = Casae()

    optimizer=torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)


    criterion.cuda()
    model.cuda()
    loss=None
    for epoch in range(100):



        for data in dset_loaders['train']:


            print(len(dset_loaders['train']))

            image=data

            # SHOW IMAGE
            # print(image)
            #
            # img = np.squeeze(image[0].numpy())
            #
            # fig = plt.figure(figsize=(5, 5))
            # ax = fig.add_subplot(111)
            # ax.imshow(img, cmap='gray')
            #
            # plt.show()
            #
            # input("PRESS ENTER")

            optimizer.zero_grad()

            in_data= image[0].cuda()


            out=model(in_data)


            loss=criterion(out,image[0].cuda())


            loss.backward()

            optimizer.step()


        print("LOSS")

        print (loss.item())

            #input("test")
















