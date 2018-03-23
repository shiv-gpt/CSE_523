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
#import cv2 as cv
import numpy as np

data_dir= '/home/shivang/Downloads/img_align_celeba/'
cur_dir = '/home/shivang/Downloads/img_resized_celebA_64/Images/'
cur_dir_w_s = '/home/shivang/Downloads/img_resized_celebA_64'

data_dir= '/home/shivang/img_align_celeba/'
cur_dir = '/home/shivang/img_resized_celebA_64/Images/'
cur_dir_w_s = '/home/shivang/img_resized_celebA_64'
img_list = os.listdir(cur_dir)
batch_size = 128
num_epochs = 50
l_r = 0.0002
beta1 = 0.5
beta2 = 0.999
z_dim = 128
n_g_c = 64
n_d_c = 64
leaky_relu_param = 0.2
image_size = 64
crop_size = 108
print("Loading Data....")

# imgs = []
# batches = []
# batch_size = 100
# c = 0
# b = 1
# arr = np.zeros((batch_size, 3, 64, 64))
# for i in img_list:  
#     new_img = scipy.misc.imread(cur_dir + i, mode='RGB')
#     arr[c,:] = new_img.T
#     c = c + 1
#     if c == 100:
#         batches.append(arr)
#         #break
#         print(b)
#         b = b + 1
#         c = 0
    # if b == 10:
    # 	break
        
#batches.append(arr[0:c, :])
    #scipy.misc.imsave(resize_dir + i, new_img)
img_list = os.listdir(data_dir)

if not os.path.isdir(cur_dir):
    os.mkdir(cur_dir)
    for i in range(10000):
    	# print(img_list[i])
    	img = Image.open(data_dir + img_list[i])
    	c_x = (img.size[0] - crop_size) // 2
    	c_y = (img.size[1] - crop_size) // 2
    	img = img.crop([c_x, c_y, c_x + crop_size, c_y + crop_size])
    	img = img.resize((image_size, image_size), Image.BILINEAR)
    	img.save(cur_dir + img_list[i], 'JPEG')
    	if i % 1000 == 0:
    		print('Resizing' + str(i) + 'images...')

transform = transforms.Compose([transforms.Scale(image_size),transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

celebA_data = datasets.ImageFolder(cur_dir_w_s, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=celebA_data, batch_size=batch_size, shuffle=True)



use_gpu = True

def ToCuda(x):
	return x.cuda()

def ToVariable(x):
	if use_gpu == True:
		return ToCuda(Variable(x))
	else: return Variable(x)
print("Loading Data Finished....")



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.e1 = nn.Conv2d(3, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)

        self.e2 = nn.Conv2d(128, 128*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128*2)

        self.e3 = nn.Conv2d(128*2, 128*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(128*4)

        self.e4 = nn.Conv2d(128*4, 128*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(128*8)

        # self.e5 = nn.Conv2d(128*8, 128*8, 4, 2, 1)
        # self.bn5 = nn.BatchNorm2d(128*8)

        self.fc1 = nn.Linear(128*8*4*4, 500)
        self.fc2 = nn.Linear(128*8*4*4, 500)

        #decoder
        self.d1 = nn.Linear(500, 128*8*2*4*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(128*8*2, 128*8, 3, 1) #128*8, 8, 8
        self.bn6 = nn.BatchNorm2d(128*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2) ##128*8, 16, 16
        self.pd2 = nn.ReplicationPad2d(1) #128*8, 18, 18
        self.d3 = nn.Conv2d(128*8, 128*4, 3, 1) ##128*4, 16, 16
        self.bn7 = nn.BatchNorm2d(128*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2) ##128*4, 32, 32
        self.pd3 = nn.ReplicationPad2d(1) ##128*4, 34, 34
        self.d4 = nn.Conv2d(128*4, 128*2, 3, 1) ##128*2, 32, 32
        self.bn8 = nn.BatchNorm2d(128*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)  ##128*2, 64, 64
        self.pd4 = nn.ReplicationPad2d(1) ##128*2, 66, 66
        self.d5 = nn.Conv2d(128*2, 3, 3, 1) ##3, 64, 64
        # self.bn9 = nn.BatchNorm2d(128, 1.e-3)

        # self.up5 = nn.UpsamplingNearest2d(scale_factor=2) 
        # self.pd5 = nn.ReplicationPad2d(1)
        # self.d6 = nn.Conv2d(128, 3, 3, 1)##3, 64, 64

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def reparametrize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()		
		eps = torch.cuda.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mu)


    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        # h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h4 = h4.view(-1, 128*8*4*4)

        return self.fc1(h4), self.fc2(h4)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, 128*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        # h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))
        h5 = self.leakyrelu(self.d5(self.pd4(self.up4(h4))))
        # h5 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        #return h5
        return self.sigmoid(h5)

        
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 3, 64, 64))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

#         self.encoder = nn.Sequential(
# #             nn.Linear(3072, 1024), 
# #             nn.ReLU(True), 
# #             nn.Linear(1024, 256), 
# #             nn.ReLU(True), 
# #             nn.Linear(256,64)
            
# #             nn.Conv2d(3, 8, 3, stride = 2, padding = 1), #8, 16, 16
# #             nn.MaxPool2d(2, stride=2), #8, 8, 8
# #             nn.LeakyReLU(0.2),
# #             nn.Conv2d(8, 16, 3, stride = 2, padding = 1), #16, 4, 4
# #             nn.MaxPool2d(2, stride = 1), #16, 3, 3
# #             #nn.Sigmoid()
# #             nn.LeakyReLU(0.2)
            
#             nn.Conv2d(3, 32, 4, stride = 2, padding = 1, bias = False), #32, 16, 16
#             nn.LeakyReLU(0.2, False),
#             nn.Conv2d(32, 64, 4, stride = 2, padding = 1, bias = False), #64, 8, 8
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, False),
#             nn.Conv2d(64, 128, 4, stride = 2, padding = 1, bias = False), #128, 4, 4
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, False),
#             nn.Conv2d(128, 128, 4, stride = 4, padding = 0, bias = False), #128, 1, 1
#             nn.Sigmoid()            
#         )
#         self.decoder = nn.Sequential(
# #             nn.Linear(64, 256),
# #             nn.ReLU(True),
# #             nn.Linear(256, 1024),
# #             nn.ReLU(True),
# #             nn.Linear(1024, 3072),
# #             nn.Tanh()
            
# #             nn.ConvTranspose2d(16, 16, 2, stride=3, padding=0), #16, 8, 8
# #             nn.LeakyReLU(0.2),
# #             nn.ConvTranspose2d(16, 8, 2, stride=2, padding=0),#8 , 16, 16
# #             nn.LeakyReLU(0.2),
# #             nn.ConvTranspose2d(8, 3, 2, stride=2, padding=0), #3, 32, 32
# #             #nn.Tanh()
# #             #nn.Hardtanh(0, 1)
# #             nn.LeakyReLU(0.2)
            
#             nn.ConvTranspose2d(128, 128, 4, 1, 0, bias = False), #128, 4, 4
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), #64, 8, 8
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False), #32, 16, 16
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 32, 4, 2, 1, bias = False), #32, 32, 32
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 3, 3, 1, 1, bias = False), #3, 32, 32
#             #nn.Hardtanh(0, 1)
#         )
		#encoder
		
  
def show(img, i):
    npimg = (img.cpu()).numpy()
    # print("npimg.shape = " + str(npimg.shape))
    #plt.imshow(np.reshape(npimg, (32,32,3)), interpolation='nearest')
    scipy.misc.imsave("Result_vae_" + str(epoch) + ".jpg" , np.transpose(npimg, (2,1,0))) #interpolation='nearest')

    #plt.imshow(npimg, interpolation='nearest')

def denorm(x):
    """Convert range (-1, 1) to (0, 1)"""
    out = (x + 1) / 2
    return out.clamp(0, 1)

    
vae = VAE()
if use_gpu:
	vae = ToCuda(vae)
#loss_calculator = nn.MSELoss()
loss_calculator = nn.BCELoss()
loss_calculator.size_average = False
adam_optimize = torch.optim.Adam(vae.parameters(), l_r, [beta1, beta2]) #""", weight_decay=1e-3"""


def loss_calc(out, inp, mean, log_var):
    bce = loss_calculator(out, inp)
    #kl_divergence = torch.sum(0.5*(mean.pow(2) + log_var.exp() - 1 - log_var))
    # kl_divergence = torch.sum(0.5*(mean.pow(2) + torch.log(log_var) - 1 - log_var))

    kl_divergence_element = mean.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    kl_divergence = torch.sum(kl_divergence_element).mul_(-0.5)

    return bce + kl_divergence



# random.shuffle(batches)
# train = batches[0:len(batches)/7]
# test = batches[len(batches)/7 : len(batches)]
print("Training Started....")


for epoch in range(num_epochs+1):
    train_loss = 0
    for d in data_loader:
        i, l = d
	i = ToVariable((torch.Tensor(i)))
	o, log_var, mean = vae(i)
	print(o.size())
	loss_val = loss_calc(o, i, mean, log_var)
	train_loss = train_loss + loss_val.data[0]
	adam_optimize.zero_grad()
	loss_val.backward()
	adam_optimize.step()
    print("Loss after Epoch " + str(epoch) + "  =  " + str(loss_val.data[0]))
    print("Total Batch Loss after Epoch " + str(epoch) + "  =  " + str(train_loss))
    if epoch%5==0:
	out = denorm(o.data)
	#out = o.data
	#print("Out.shape = " + str(out.shape))
	out = out.view(out.size(0), 3, 64, 64)
	#print("Out.shape after = " + str(out.shape))
	show(make_grid(out, nrow=10), epoch)

torch.save(vae.state_dict(), './celeb_64_vae.pth')
