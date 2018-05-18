from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import math
# our data loader
import MultipieLoader
import gc
import numpy as np
from model import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

# my functions
#import zx

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--cuda', default = True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu_ids', type=int, default=0, help='ids of GPUs to use')
parser.add_argument('--modelPath', default='', help="path to model (to continue training)")
parser.add_argument('--dirCheckpoints', default='.', help='folder to model checkpoints')
parser.add_argument('--dirImageoutput', default='.', help='folder to output images')
parser.add_argument('--dirTestingoutput', default='.', help='folder to testing results/images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--epoch_iter', type=int,default=200, help='number of epochs on entire dataset')
parser.add_argument('--location', type = int, default=0, help ='where is the code running')
parser.add_argument('--margin', type = float, default=0.5, help ='Marging for the triplet loss')
parser.add_argument('-f',type=str,default= '', help='dummy input required for jupyter notebook')
parser.add_argument('--isTraining', type=int,default= 1, help='Training or Testing')
opt = parser.parse_args()
print(opt)


## do not change the data directory
opt.data_dir_prefix = '/nfs/bigdisk/zhshu/data/fare/'

## change the output directory to your own
opt.output_dir_prefix = './'
opt.dirCheckpoints    = opt.output_dir_prefix + '/checkpoints/LightingTripletTest'
opt.dirImageoutput    = opt.output_dir_prefix + '/images/LightingTripletTest'
opt.dirTestingoutput  = opt.output_dir_prefix + '/testing/LightingTripletTest'

opt.imgSize = 64



try:
    os.makedirs(opt.dirCheckpoints)
except OSError:
    pass
try:
    os.makedirs(opt.dirImageoutput)
except OSError:
    pass
try:
    os.makedirs(opt.dirTestingoutput)
except OSError:
    pass


# sampe iamges
# def visualizeAsImages(img_list, output_dir, 
#                       n_sample=4, id_sample=None, dim=-1, 
#                       filename='myimage', nrow=2, 
#                       normalize=False):
#     if id_sample is None:
#         images = img_list[0:n_sample,:,:,:]
#     else:
#         images = img_list[id_sample,:,:,:]
#     if dim >= 0:
#         images = images[:,dim,:,:].unsqueeze(1)
#     vutils.save_image(images, 
#         '%s/%s'% (output_dir, filename+'.png'),
#         nrow=nrow, normalize = normalize, padding=2)

def visualizeAsImages(img_list, output_dir, 
                      n_sample=4, id_sample=None, dim=-1, 
                      filename='myimage', nrow=2, 
                      normalize=False):
    if id_sample is None:
        images = img_list[0:n_sample,:,:,:]
    else:
        images = img_list[id_sample,:,:,:]
    if dim >= 0:
        images = images[:,dim,:,:].unsqueeze(1)
    images.sub_(0.5).div_(0.5)
    vutils.save_image(images, 
        '%s/%s'% (output_dir, filename+'.png'),
        nrow=nrow, padding=2)


def parseSampledDataTripletMultipie(dp0_img,  dp9_img, dp1_img):
    ###
    dp0_img  = dp0_img.float()/255 # convert to float and rerange to [0,1]
    dp0_img  = dp0_img.permute(0,3,1,2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
    ###
    dp9_img  = dp9_img.float()/255 # convert to float and rerange to [0,1]
    dp9_img  = dp9_img.permute(0,3,1,2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
    ###
    dp1_img  = dp1_img.float()/255 # convert to float and rerange to [0,1]
    dp1_img  = dp1_img.permute(0,3,1,2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
    return dp0_img, dp9_img, dp1_img


def setFloat(*args):
    barg = []
    for arg in args: 
        barg.append(arg.float())
    return barg

def setCuda(*args):
    barg = []
    for arg in args: 
        barg.append(arg.cuda())
    return barg

def setAsVariable(*args):
    barg = []
    for arg in args: 
        barg.append(Variable(arg))
    return barg    

def setAsDumbVariable(*args):
    barg = []
    for arg in args: 
        barg.append(Variable(arg,requires_grad=False))
    return barg   




# Training data folder list
TrainingData = []
#session 01
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_01_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_02_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_03_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_04_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_05_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_06_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_07_select/')

#session 02
"""
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_01_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_02_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_03_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_04_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_05_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_06_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session02_07_select/')
#session 03
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_01_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_02_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_03_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_04_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session03_05_select/')
#session 04
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_01_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_02_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_03_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_04_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_05_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_06_select/')
TrainingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session04_07_select/')
"""

# Testing
TestingData = []
TestingData.append(opt.data_dir_prefix + 'real/multipie_select_batches/session01_select_test/')


m = lightingModel(zdim=512)
m = m.cuda()
# print(m.parameters())
trainer = Trainer(m, params=opt)
counter = []
loss_history = []

def show_plot(iteration,loss, filepath):
    print(loss)
    plt.plot(iteration,loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    # plt.show()
    plt.savefig(filepath + '/loss_plot.png')

def getTestingSample():
    dataroot = random.sample(TestingData,1)[0]
    dataset = MultipieLoader.FareMultipieLightingTripletsFrontal(opt, root=dataroot, resize=64)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    for batch_idx, data_point in enumerate(dataloader, 0):
        dp0_img, dp9_img, dp1_img = data_point
        dp0_img, dp9_img, dp1_img = parseSampledDataTripletMultipie(dp0_img, dp9_img, dp1_img)
        if opt.cuda:
            dp0_img, dp9_img, dp1_img = setCuda(dp0_img, dp9_img, dp1_img)
        dp0_img, dp9_img, dp1_img = setAsVariable(dp0_img, dp9_img, dp1_img)
        return dp0_img

def testModel():
    def parse_image(dp0_img):
        print("before ========= ")
        print(dp0_img.size())
        dp0_img  = dp0_img.float()/255 # convert to float and rerange to [0,1]
        dp0_img  = dp0_img.permute(0,3,1,2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
        # dp0_img_1 = np.array(dp0_img).reshape(64,64,3)
        print("after ========= ")
        print(dp0_img.size())
        return dp0_img

    def save_images(imgs, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        c = 0
        for i in imgs:
            img = Image.open(i)
            img = img.convert('RGB')
            img = img.resize((64, 64),Image.ANTIALIAS)
            # img0 = np.array(img0)
            img.save(path + str(c) + ".png")
            c = c+1

    def load_images(path,batch_size):
        ret = torch.ones(5,64,64,3)
        for i in range(batch_size):
            img = Image.open(path + str(i) + ".png")
            img = img.convert('RGB')
            img = img.resize((64, 64),Image.ANTIALIAS)
            img0 = np.array(img)
            ret[i,:,:,:] = torch.from_numpy(img0)
        # data = dset.ImageFolder(path, transform=transforms.ToTensor())
        # dl = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size)
        # for d in dl:
        #     i,l = d
        #     # img = Image.open(path + str(i) + ".png")
        #     images.append(parse_image(i))
        return parse_image(ret)

    def set_cuda(x):
        return x.cuda()

    # def getitem(d, ids, e, p, l):
    #     # different ids, same ide, same idl, same idt, same idp
    #     print(len(d.ids))
    #     for i in range(len(d.ids)):
    #         if(d.ids[i] == ids and d.ide[i] == e and d.idp[i] == p and d.idl[i] == l):
    #             return d.imgs[i]

    dataroot0 = TrainingData[0]
    dataset0 = MultipieLoader.FareMultipieLightingTripletsFrontal(opt, root=dataroot0, resize=64)
    dataset0.get_Sample('params0.csv')

    dataroot5 = TrainingData[5]
    dataset5 = MultipieLoader.FareMultipieLightingTripletsFrontal(opt, root=dataroot5, resize=64)
    dataset5.get_Sample('params5.csv')

    

    # input_img_list = [dataset0.getitem('001','01','051','00'), dataset0.getitem('002','01','051','00'), dataset0.getitem('003','01','051','00'), dataset0.getitem('004','01','051','00'), dataset0.getitem('005','01','051','00')]
    # print(input_img_list)
    # # i = []
    # save_images(input_img_list, "test_images/inputs/")
    # # for img in input_img_list:
    # #     i.append(parse_image(img))
    # # dataroot5 = TrainingData[5]
    # # dataset5 = MultipieLoader.FareMultipieLightingTripletsFrontal(opt, root=dataroot5, resize=64)
    # # dataset5.get_Sample('params5.csv')
    # swap_img_list = [dataset5.getitem('028','02','051','12'), dataset5.getitem('028','02','051','12'), dataset5.getitem('028','02','051','12'), dataset5.getitem('028','02','051','12'), dataset5.getitem('028','02','051','12')]
    # save_images(swap_img_list, "test_images/swap_inputs/")

    # target_img_list = [dataset5.getitem('001','01','051','12'), dataset5.getitem('002','01','051','12'), dataset5.getitem('003','01','051','12'), dataset5.getitem('004','01','051','12'), dataset5.getitem('005','01','051','12')]
    # # print(target_img_list)
    # save_images(target_img_list, "test_images/targets/")

    inputs = Variable(set_cuda(load_images("test_images/inputs/1/", 5)))
    input_swap = Variable(set_cuda(load_images("test_images/swap_inputs/1/", 5)))
    targets = Variable(set_cuda(load_images("test_images/targets/1/",5)))

    visualizeAsImages(inputs.data.clone(), 
        opt.dirImageoutput, 
        filename='test_input_iter__img0', n_sample = 5, nrow=1, normalize=False)
    visualizeAsImages(input_swap.data.clone(), 
        opt.dirImageoutput, 
        filename='test_input_swap_orig_iter__img0', n_sample = 5, nrow=1, normalize=False)

    model = torch.load("models/783a5ffc-7a8e-4fa7-b67d-0e92d41fdc40/model.pth")
    # model.eval()
    lightCode_i, nonLightCode_i, o_i = model.forward(inputs)
    lightCode_i_s, nonLightCode_i_s, o_i_s = model.forward(input_swap)
    # lightCode_t, nonLightCode_t, o_t = model.forward(targets)

    z = torch.cat([lightCode_i_s, nonLightCode_i],1)
    z = z.unsqueeze(2)
    z = z.unsqueeze(3)
    o_swap = model.D(z)
    # print(lightCode0.size())
    
    visualizeAsImages(o_i.data.clone(), 
        opt.dirImageoutput, 
        filename='test_ouput_iter__img0', n_sample = 5, nrow=1, normalize=False)
    visualizeAsImages(o_swap.data.clone(), 
        opt.dirImageoutput, 
        filename='test_ouput_swap_iter__img0', n_sample = 5, nrow=1, normalize=False)
    visualizeAsImages(targets.data.clone(), 
        opt.dirImageoutput, 
        filename='test_target_iter__img0', n_sample = 5, nrow=1, normalize=False)


    # swap_input = load_images("test_images/swap_inputs/", 1)

    # dataloader0 = torch.utils.data.DataLoader(dataset0, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    # for batch_idx, data_point in enumerate(dataloader0, 0):
    #     # dp0_img, dp9_img, dp1_img = data_point
    #     # dp0_img, dp9_img, dp1_img = parseSampledDataTripletMultipie(dp0_img, dp9_img, dp1_img)
    #     # if opt.cuda:
    #     #     dp0_img, dp9_img, dp1_img = setCuda(dp0_img, dp9_img, dp1_img)
    #     # dp0_img, dp9_img, dp1_img = setAsVariable(dp0_img, dp9_img, dp1_img)
    #     print(batch_idx)
    #     break
        # return dp0_img


# ------------ training ------------ #
if opt.isTraining == 1:
    doTraining = True
    doTesting = False
else:
    doTraining = False
    doTesting = True
iter_mark=0
for epoch in range(opt.epoch_iter):
    if doTraining:
        loss_sum = 0
        train_amount = 0+1e-6
        gc.collect() # collect garbage
        # for subprocid in range(10):
        # random sample a dataroot
        dataroot = random.sample(TrainingData,1)[0]
        aaaa=0
        dataset = MultipieLoader.FareMultipieLightingTripletsFrontal(opt, root=dataroot, resize=64)
        print('# size of the current (sub)dataset is %d' %len(dataset))
        train_amount = train_amount + len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
        for batch_idx, data_point in enumerate(dataloader, 0):

            aaaa +=1
            if aaaa>20:
                break

            gc.collect() # collect garbage
            # sample the data points: 
            # dp0_img: image of data point 0
            # dp9_img: image of data point 9, which is different in lighting compare to dp0
            # dp1_img: image of data point 1, which is different in person compare to dp1
            dp0_img, dp9_img, dp1_img = data_point
            dp0_img, dp9_img, dp1_img = parseSampledDataTripletMultipie(dp0_img, dp9_img, dp1_img)
            if opt.cuda:
                dp0_img, dp9_img, dp1_img = setCuda(dp0_img, dp9_img, dp1_img)
            dp0_img, dp9_img, dp1_img = setAsVariable(dp0_img, dp9_img, dp1_img )
            

            # print(dp0_img.size())
            # print(dp1_img.size())
            # print(dp9.img.size())
            #############################
            ## put training code here ###
            #############################
            loss = trainer.ae_step(dp0_img, dp1_img, dp9_img)
            loss_sum = loss_sum + loss.data[0]
            if epoch%5==0:
            	print("Epoch = " + str(epoch))
            	print("Loss = " + str(loss))
            	print("")

            # print(dp0_img.size())
            # print(dp9_img.size())
            # print(dp1_img.size())
            # visualizeAsImages(dp0_img.data.clone(), 
            #     opt.dirImageoutput, 
            #     filename='iter_'+str(iter_mark)+'_img0', n_sample = 25, nrow=5, normalize=False)
            # visualizeAsImages(dp9_img.data.clone(), 
            #     opt.dirImageoutput, 
            #     filename='iter_'+str(iter_mark)+'_img9', n_sample = 25, nrow=5, normalize=False)
            # visualizeAsImages(dp1_img.data.clone(), 
            #     opt.dirImageoutput, 
            #     filename='iter_'+str(iter_mark)+'_img1', n_sample = 25, nrow=5, normalize=False)

            # print('Test image saved, kill the process by Ctrl + C')
    # gc.collect() # collect garbage
        counter.append(epoch)
        loss_history.append(loss_sum)
# show_plot(counter, loss_history)
if(doTraining):
    import uuid
    directory = str(uuid.uuid4())
    while os.path.exists("/models/" + directory) is True:
    	directory = str(uuid.uuid4())
    os.makedirs("models/" + directory)
    print("Saved in ==============  " + directory)
    show_plot(counter, loss_history, "models/" + directory)
    torch.save(trainer.ae, "models/" + directory + "/model.pth")
elif (doTesting):
    # testModel()
    # random sample a dataroot
    dataroot = random.sample(TrainingData,1)[0]
    # aaaa=0
    dataset = MultipieLoader.FareMultipieLightingTripletsFrontal(opt, root=dataroot, resize=64)
    print('# size of the current (sub)dataset is %d' %len(dataset))
    # train_amount = train_amount + len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    # dataset.get_Sample()
    for batch_idx, data_point in enumerate(dataloader, 0):
        dp0_img, dp9_img, dp1_img = data_point
        dp0_img, dp9_img, dp1_img = parseSampledDataTripletMultipie(dp0_img, dp9_img, dp1_img)
        if opt.cuda:
            dp0_img, dp9_img, dp1_img = setCuda(dp0_img, dp9_img, dp1_img)
        dp0_img, dp9_img, dp1_img = setAsVariable(dp0_img, dp9_img, dp1_img )
        print(dp0_img.size())
        print(dp9_img.size())
        print(dp1_img.size())
        visualizeAsImages(dp0_img.data.clone(), 
            opt.dirImageoutput, 
            filename='test_input_iter__img0', n_sample = 25, nrow=5, normalize=False)
        model = torch.load("models/783a5ffc-7a8e-4fa7-b67d-0e92d41fdc40/model.pth")
        lightCode0, nonLightCode0, o0 = model.forward(dp0_img)
        print("######################")
        print(o0.size())
        print(lightCode0.size())
        # lightCode1, nonLightCode1, o1 = model.forward(dp1_img)
        # lightCode9, nonLightCode9, o9 = model.forward(dp9_img)
        dp0_img_test = getTestingSample()
        lightCode_swap, nonLightCode_swap, o_swap_test = model.forward(dp0_img_test)
        z = torch.cat([lightCode_swap, nonLightCode0],1)
        z = z.unsqueeze(2)
        z = z.unsqueeze(3)
        o_swap = model.D(z)
        visualizeAsImages(o0.data.clone(), 
            opt.dirImageoutput, 
            filename='test_ouput_iter__img0', n_sample = 25, nrow=5, normalize=False)
        visualizeAsImages(o_swap.data.clone(), 
            opt.dirImageoutput, 
            filename='test_ouput_swap_iter__img0', n_sample = 25, nrow=5, normalize=False)
        visualizeAsImages(dp0_img_test.data.clone(), 
            opt.dirImageoutput, 
            filename='test_input_swap_orig_iter__img0', n_sample = 25, nrow=5, normalize=False)
        break
# ------------ testing ------------ #























    ##
