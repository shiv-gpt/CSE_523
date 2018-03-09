import os
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
import torch.utils.data as data_utils
from torchvision.utils import make_grid
import numpy as np
import scipy
import scipy.misc
import random

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import cv2 as cv
import numpy as np

cur_dir = '/home/shivang/celebA_64/'
img_list = os.listdir(cur_dir)


print("Loading Data....")

imgs = []
batches = []
batch_size = 100
c = 0
b = 1
arr = np.zeros((batch_size, 3, 64, 64))
for i in img_list:  
    new_img = scipy.misc.imread(cur_dir + i, mode='RGB')
    arr[c,:] = new_img.T
    c = c + 1
    if c == 100:
        batches.append(arr)
        #break
        print(b)
        b = b + 1
        c = 0
    # if b == 10:
    # 	break
        
#batches.append(arr[0:c, :])
    #scipy.misc.imsave(resize_dir + i, new_img)


print("Loading Data Finished....")

num_epochs = 50
batch_size = 100
l_r = 0.0001

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.d = nn.Sequential(
#             nn.Linear(784, 392), 
#             nn.LeakyReLU(0.2), 
#             nn.Linear(392, 256), 
#             nn.LeakyReLU(0.2), 
#             nn.Linear(256,1), 
#             nn.Sigmoid()
            nn.Conv2d(3, 32, 4, stride = 2, padding = 1, bias = False), #32, 32, 32
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(32, 64, 4, stride = 2, padding = 1, bias = False), #64, 16, 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(64, 128, 4, stride = 2, padding = 1, bias = False), #128, 8, 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(128, 256, 4, stride = 2, padding = 1, bias = False), #256, 4, 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(256, 1, 4, stride = 4, padding = 0, bias = False), #1, 1, 1
            nn.Sigmoid() 
        )
    def forward(self, x):
        return self.d(x)

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.g = nn.Sequential(
#             nn.Linear(64, 128), 
#             nn.LeakyReLU(0.2), 
#             nn.Linear(128, 392), 
#             nn.LeakyReLU(0.2), 
#             nn.Linear(392,784), 
#             nn.Tanh()
            
#             nn.ConvTranspose2d(128, 128, 4, 1, 0, bias = False), #128, 4, 4
            # 
            nn.ConvTranspose2d(100, 128*8, 4, 1, 0, bias = False), #1024, 4, 4
            nn.BatchNorm2d(128*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(128*8, 128*4, 4, 2, 1, bias = False), #512, 8, 8
            nn.BatchNorm2d(128*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(128*4, 128*2, 4, 2, 1, bias = False), #256, 16, 16
            nn.BatchNorm2d(128*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128*2, 128, 4, 2, 1, bias = False), #128, 32, 32
            #nn.Tanh()
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias = False), #3, 64, 64
            nn.Tanh()
        )
    def forward(self, x):
        return self.g(x)
    
def show(img, i):
    npimg = (img.cpu()).numpy()
    # print("npimg.shape = " + str(npimg.shape))
    #plt.imshow(np.reshape(npimg, (32,32,3)), interpolation='nearest')
    scipy.misc.imsave("Result_" + str(epoch) + ".jpg" , np.transpose(npimg, (2,1,0))) #interpolation='nearest')



gen = G()
gen.cuda()
dis = D()
dis.cuda()
loss_calculator = nn.BCELoss()
adam_optimize_gen = torch.optim.Adam(gen.parameters(), lr=l_r)
adam_optimize_dis = torch.optim.Adam(dis.parameters(), lr=l_r)



random.shuffle(batches)
train = batches[0: (7 * len(batches))/10]
test = batches[(7 * len(batches))/10 : len(batches)]


print("Training Started....")

print(str(len(train)))
for epoch in range(num_epochs+1):
    gen_output = np.zeros((64, 64, 3))
    for d in train:
        i = d
        # print(i.shape)
        i = Variable((torch.Tensor(i)).cuda())
        # print("Training inside....")
        
#         real_label = Variable(torch.ones(batch_size, 128, 1, 1))
#         fake_label = Variable(torch.zeros(batch_size, 128, 1, 1))
        real_label = Variable((torch.ones(batch_size, 1, 1, 1)).cuda())
        fake_label = Variable((torch.zeros(batch_size, 1, 1, 1)).cuda())
        
        #training of discriminator
        o = dis(i)
        # print(o.shape)
        dis_loss_val_r = loss_calculator(o, real_label)
        
        fake_input = Variable((torch.randn(batch_size, 100, 1, 1)).cuda())
        fake_image = gen(fake_input)
        o = dis(fake_image)
        dis_loss_val_f = loss_calculator(o, fake_label)
        
        total_dis_loss = dis_loss_val_r + dis_loss_val_f
        dis.zero_grad()
        total_dis_loss.backward()
        adam_optimize_dis.step()
        
        
#         real_label = Variable(torch.ones(batch_size, 128, 1, 1))
        real_label = Variable((torch.ones(batch_size, 1, 1, 1)).cuda())
        fake_input = Variable((torch.randn(batch_size, 100, 1, 1)).cuda())
        gen_output = gen(fake_input)
        o = dis(gen_output)
        
        loss_val_g = loss_calculator(o, real_label) ##as we have to assume that the images genrated by the generator are real and have successfully fooled the discriminator
        
        gen.zero_grad()
        loss_val_g.backward()
        adam_optimize_gen.step()
        
    print("Loss Gen after Epoch " + str(epoch) + "  =  " + str(loss_val_g.data[0]))
    print("Loss Dis after Epoch " + str(epoch) + "  =  " + str(total_dis_loss.data[0]))
    if epoch%5==0:
        out = gen_output.data
        # print("Out.shape = " + str(out.shape))
        out = out.view(out.size(0), 3, 64, 64)
        # print("Out.shape after = " + str(out.shape))
        # show(make_grid(out, nrow=10,normalize=True), epoch)
        show(make_grid(out, nrow=10), epoch)


print("Training Finished....")
        
torch.save(gen.state_dict(), './gen_conv_celebA.pth')
torch.save(dis.state_dict(), './dis_conv_celebA.pth')