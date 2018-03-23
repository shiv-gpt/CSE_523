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
from PIL import Image

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
leaky_relu_param = 0.05
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
#     # if b == 10:
#     # 	break
        
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
            nn.Conv2d(3, n_g_c, 4, stride = 2, padding = 1, bias = False), #64, 32, 32
            nn.LeakyReLU(leaky_relu_param, False),
            nn.Conv2d(n_g_c, n_g_c*2, 4, stride = 2, padding = 1, bias = False), #128, 16, 16
            nn.BatchNorm2d(n_g_c*2),
            nn.LeakyReLU(leaky_relu_param, False),
            nn.Conv2d(n_g_c*2, n_g_c*4, 4, stride = 2, padding = 1, bias = False), #256, 8, 8
            nn.BatchNorm2d(n_g_c*4),
            nn.LeakyReLU(leaky_relu_param, False),
            nn.Conv2d(n_g_c*4, n_g_c*8, 4, stride = 2, padding = 1, bias = False), #512, 4, 4
            nn.BatchNorm2d(n_g_c*8),
            nn.LeakyReLU(leaky_relu_param, False),
            nn.Conv2d(n_g_c*8, 1, 4, stride = 1, padding = 0, bias = False), #1, 1, 1 
        )
    def forward(self, x):
    	#return self.d(x)
        return self.d(x).squeeze()

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
            nn.ConvTranspose2d(z_dim, n_g_c*8, 4, 1, 0, bias = False), #512, 4, 4
            # nn.BatchNorm2d(n_g_c*8),
            nn.LeakyReLU(leaky_relu_param, False),
            nn.ConvTranspose2d(n_g_c*8, n_g_c*4, 4, 2, 1, bias = False), #256, 8, 8
            nn.BatchNorm2d(n_g_c*4),
            nn.LeakyReLU(leaky_relu_param, False),
            nn.ConvTranspose2d(n_g_c*4, n_g_c*2, 4, 2, 1, bias = False), #128, 16, 16
            nn.BatchNorm2d(n_g_c*2),
            nn.LeakyReLU(leaky_relu_param, False),
            nn.ConvTranspose2d(n_g_c*2, n_g_c, 4, 2, 1, bias = False), #64, 32, 32
            #nn.Tanh()
            nn.BatchNorm2d(n_g_c),
            nn.LeakyReLU(leaky_relu_param, False),
            nn.ConvTranspose2d(n_g_c, 3, 4, 2, 1, bias = False), #3, 64, 64
            nn.Tanh()
        )
    def forward(self, x):
        return self.g(x)
    
def show(img, i):
    npimg = (img.cpu()).numpy()
    # print("npimg.shape = " + str(npimg.shape))
    #plt.imshow(np.reshape(npimg, (32,32,3)), interpolation='nearest')
    scipy.misc.imsave("Result_" + str(epoch) + ".jpg" , np.transpose(npimg, (2,1,0))) #interpolation='nearest')

def denorm(x):
        """Convert range (-1, 1) to (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

print("Loading Data Finished 2  2 ....")

gen = G()
if use_gpu:
	gen = ToCuda(gen)
dis = D()
if use_gpu:
	dis = ToCuda(dis)
loss_calculator = nn.BCELoss()
adam_optimize_gen = torch.optim.Adam(gen.parameters(), l_r, [beta1, beta2])
adam_optimize_dis = torch.optim.Adam(dis.parameters(), l_r, [beta1, beta2])
print("Loading Data Finished 3 3 3 3....")


# random.shuffle(batches)
# train = batches[0: (7 * len(batches))/10]
# test = batches[(7 * len(batches))/10 : len(batches)]


print("Training Started....")

# print(str(len(train)))
for epoch in range(num_epochs+1):
    gen_output = np.zeros((64, 64, 3))
    for d in data_loader:
        i, l = d
        i = ToVariable((torch.Tensor(i)))
        # i = Variable((torch.Tensor(i)).cuda())
        # print("Training inside....")
        
#         real_label = Variable(torch.ones(batch_size, 128, 1, 1))
#         fake_label = Variable(torch.zeros(batch_size, 128, 1, 1))
        # real_label = Variable((torch.ones(batch_size, 1, 1, 1)))
        # fake_label = Variable((torch.zeros(batch_size, 1, 1, 1)))
        # real_label = Variable((torch.ones(batch_size, 1, 1, 1)).cuda())
        # fake_label = Variable((torch.zeros(batch_size, 1, 1, 1)).cuda())
        real_label = ToVariable(torch.ones(batch_size, 1, 1, 1))
        fake_label = ToVariable(torch.zeros(batch_size, 1, 1, 1))
        
        #training of discriminator
        o = dis(i)
        # print(o.shape)
        # dis_loss_val_r = loss_calculator(o, real_label)
        dis_loss_val_r = torch.mean((o - 1) ** 2)


        # fake_input = Variable((torch.randn(batch_size, 100, 1, 1))""".cuda()""")
        fake_input = ToVariable(torch.randn(batch_size, z_dim, 1, 1))
        # fake_input = Variable((torch.randn(batch_size, z_dim, 1, 1)).cuda())
        fake_image = gen(fake_input)
        o = dis(fake_image)
        # dis_loss_val_f = loss_calculator(o, fake_label)
        dis_loss_val_f = torch.mean(o ** 2)

        total_dis_loss = dis_loss_val_r + dis_loss_val_f
        dis.zero_grad()
        gen.zero_grad()
        total_dis_loss.backward()
        adam_optimize_dis.step()
        
        
#         real_label = Variable(torch.ones(batch_size, 128, 1, 1))
        real_label = ToVariable(torch.ones(batch_size, 1, 1, 1))
        fake_input = ToVariable(torch.randn(batch_size, z_dim, 1, 1))
        # real_label = Variable((torch.ones(batch_size, 1, 1, 1)).cuda())
        # fake_input = Variable((torch.randn(batch_size, z_dim)).cuda())

        gen_output = gen(fake_input)
        o = dis(gen_output)
        
        loss_val_g = torch.mean((o - 1) ** 2) ##as we have to assume that the images genrated by the generator are real and have successfully fooled the discriminator
        #print(str(loss_val_g))
        dis.zero_grad()
        gen.zero_grad()
        loss_val_g.backward()
        adam_optimize_gen.step()
        
    print("Loss Gen after Epoch " + str(epoch) + "  =  " + str(loss_val_g.data[0]))
    print("Loss Dis after Epoch " + str(epoch) + "  =  " + str(total_dis_loss.data[0]))
    if epoch%5==0:
        out = denorm(gen_output.data)
        # print("Out.shape = " + str(out.shape))
        out = out.view(out.size(0), 3, 64, 64)
        # print("Out.shape after = " + str(out.shape))
        # show(make_grid(out, nrow=10,normalize=True), epoch)
        show(make_grid(out, nrow=10), epoch)


print("Training Finished....")
        
torch.save(gen.state_dict(), './gen_conv_celebA.pth')
torch.save(dis.state_dict(), './dis_conv_celebA.pth')
