import torch
import torch.nn as nn

import torch.nn.functional as F

class Casae(nn.Module):

    def __init__(self):

        super(Casae, self).__init__()

        # 1 CONVOLUTION
        self.layer1_1=nn.Sequential(nn.Conv2d(1,64,3,padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU())

        self.layer1_2 = nn.Sequential(nn.Conv2d(64, 64, 3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        self.maxpool1  = nn.MaxPool2d(2,stride=2,return_indices=True)


        # 2 CONVOLUTION
        self.layer2_1 = nn.Sequential(nn.Conv2d(64, 128, 3,dilation=2,padding=2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())

        self.layer2_2 = nn.Sequential(nn.Conv2d(128, 128, 3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())

        self.maxpool2 = nn.MaxPool2d(2, stride=2,return_indices=True)

        # 3 CONVOLUTION

        self.layer3_1 = nn.Sequential(nn.Conv2d(128, 256, 3,dilation=2,padding=2),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        self.layer3_2 = nn.Sequential(nn.Conv2d(256, 256, 3,padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        self.maxpool3 = nn.MaxPool2d(2, stride=2,return_indices=True)


        # 4 CONVOLUTION

        self.layer4_1 = nn.Sequential(nn.Conv2d(256,512, 3,dilation=4,padding=4),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        self.layer4_2 = nn.Sequential(nn.Conv2d(512, 512, 3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        self.maxpool4 = nn.MaxPool2d(2, stride=2,return_indices=True)


        # 5 CONVOLUTION

        self.layer5_1 = nn.Sequential(nn.Conv2d(512, 1024, 3,dilation=4,padding=4),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU())

        self.layer5_2 = nn.Sequential(nn.Conv2d(1024, 1024, 3,padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU())

        #self.unmaxpool5 = nn.MaxUnpool2d(2, stride=2)

        self.upsample5=nn.Upsample(scale_factor=2,mode='nearest')


        # 6 CONVOLUTION

        self.layer6_1 = nn.Sequential(nn.Conv2d(1024, 512, 3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        self.layer6_2 = nn.Sequential(nn.Conv2d(512, 512, 3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        #self.unmaxpool6 = nn.MaxUnpool2d(2, stride=2)
        self.upsample6 = nn.Upsample(scale_factor=2, mode='nearest')

        #7 CONVOLUTION

        self.layer7_1 = nn.Sequential(nn.Conv2d(512, 256, 3,padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        self.layer7_2 = nn.Sequential(nn.Conv2d(256, 256, 3,padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        #self.unmaxpool7 = nn.MaxUnpool2d(2, stride=2,padding=1)
        self.upsample7 = nn.Upsample(scale_factor=2, mode='nearest')


        # 8 CONVOLUTION

        self.layer8_1 = nn.Sequential(nn.Conv2d(256, 128, 3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())

        self.layer8_2 = nn.Sequential(nn.Conv2d(128, 128, 3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())

        #self.unmaxpool8 = nn.MaxUnpool2d(2, stride=2)
        self.upsample8 = nn.Upsample(scale_factor=2, mode='nearest')


        # 9 CONVOLUTION
        self.layer9_1 = nn.Sequential(nn.Conv2d(128, 64, 3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        self.layer9_2 = nn.Sequential(nn.Conv2d(64, 1, 1))

        #self.softmax=nn.Softmax(dim=0)


    def forward(self,x):


        x1=self.layer1_1(x)
        x1=self.layer1_2(x1)
        x1,indeces1=self.maxpool1(x1)

        print(x.size())
        print(x1.size())


        x2=self.layer2_1(x1)
        x2=self.layer2_2(x2)
        x2,indeces2 = self.maxpool2(x2)

        print(x2.size())

        x3=self.layer3_1(x2)
        x3=self.layer3_2(x3)
        x3,indeces3 = self.maxpool3(x3)

        print(x3.size())


        x4=self.layer4_1(x3)
        x4=self.layer4_2(x4)
        x4,indeces4 = self.maxpool4(x4)

        print(x4.size())


        x5=self.layer5_1(x4)
        x5=self.layer5_2(x5)
        x5 = self.upsample5(x5)

        x6=self.layer6_1(x5)
        x6=self.layer6_2(x6)
        #x6 = self.unmaxpool6(x6,indeces3,output_size=x3.size())
        x6=self.upsample6(x6)

        x7=self.layer7_1(x6)
        x7=self.layer7_2(x7)
        #x7 = self.unmaxpool7(x7,indeces2,output_size=x2.size())
        x7=self.upsample7(x7)

        x8=self.layer8_1(x7)
        x8=self.layer8_2(x8)
        #x8 = self.unmaxpool8(x8,indeces1,output_size=x1.size())
        x8=self.upsample8(x8)


        x9=self.layer9_1(x8)



        x9=self.layer9_2(x9)


        # se uso la cross entropy utilizza gia Softmax

        #x9=self.softmax(x9.view(-1)).view(1,1,512,512)


        return x9



if __name__=='__main__':


    #in_vec=torch.rand((1,1,512,512))





    # test softmax


    in_vec= torch.rand((1,1,4,4))


    #-1 = 3
    softmax= torch.nn.Softmax(dim= 3)


    softmax2=torch.nn.Softmax()


    print("TENSOR")
    out=softmax(in_vec)

    out2=softmax2(in_vec.view(4,4)).view(1,1,4,4)




    print(out)

    print(out2)





    #test=torch.rand((1,512,512))

    #torch_vec= torch.Tensor([[[1,2,3,4],[65,12,224,43],[14,520,631,124],[13,14,15,16]]])



    #print(torch_vec.size())



    #softmax= nn.Softmax2d()

    #newvector= torch_vec.view(torch_vec.size(0),-1)


    #out= softmax(in_vec)

    #print(out)

    #criterion=nn.CrossEntropyLoss()


    #in_vec2=torch.rand((1,1,512,512))



    #model=AutoConv2d()

    #optimizer = torch.optim.SGD(model.parameters(),lr=0.001)

    # out=model(inv)
    #
    #
    # print(out.size())
    #
    # print(out[0][0][300][300])
    #
    # print(out.sum(dim=0).size())
    #


    #out=model(in_vec)


    #loss=criterion(out,test.long())


    print("loss")

    #print(loss)

    #loss.backward()



    #optimizer.step()


    # x = Variable(torch.randn(3, 1, 10, 10))
    # softmax = nn.Softmax(dim=0)
    # y = softmax(x.view(-1)).view(3, 1, 10, 10)


    print(out.size())
    print(out.sum())


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))

        return x



