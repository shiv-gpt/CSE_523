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

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)

class Encoder(nn.Module):
    def __init__(self, ngc=64, leaky_relu_param=0.05, zdim=1024):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
        	nn.Conv2d(3, ngc, 4, stride = 2, padding = 1, bias = False), #ngc,32,32
        	nn.LeakyReLU(leaky_relu_param, False),
        	nn.Conv2d(ngc, ngc*2, 4, stride = 2, padding = 1, bias = False), #ngc*2, 16, 16
        	nn.BatchNorm2d(ngc*2),
        	nn.LeakyReLU(leaky_relu_param, False),
        	nn.Conv2d(ngc*2, ngc*4, 4, stride = 2, padding = 1, bias = False), #ngc*4, 8, 8
        	nn.BatchNorm2d(ngc*4),
        	nn.LeakyReLU(leaky_relu_param, False),
        	nn.Conv2d(ngc*4, ngc*8, 4, stride = 2, padding = 1, bias = False), #ngc*8, 4, 4
        	nn.BatchNorm2d(ngc*8),
        	nn.LeakyReLU(leaky_relu_param, False),
        	Flatten(),
        	# nn.Sigmoid(),       	
        	# nn.BatchNorm2d(n_g_c*32),
        	# nn.LeakyReLU(leaky_relu_param, False),
        	# nn.Conv2d(ngc*32, zdim, 4, stride = 1, padding = 0, bias = False), #zdim, 1, 1
        	)

        self.lightCode = nn.Linear(ngc*8*4*4, zdim/2)
        self.poseCode = nn.Linear(ngc*8*4*4, zdim/2)

    def forward(self, x):
    	x = self.enc(x)
    	return self.lightCode(x), self.poseCode(x)


class Decoder(nn.Module):
    def __init__(self, ngc=64, zdim=1024):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(zdim, ngc*8, 4, 1, 0, bias = False), #ngc*8, 4, 4
            nn.BatchNorm2d(ngc*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngc*8, ngc*4, 4, 2, 1, bias = False), #ngc*4, 8, 8
            nn.BatchNorm2d(ngc*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngc*4, ngc*2, 4, 2, 1, bias = False), #ngc*2, 16, 16
            nn.BatchNorm2d(ngc*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngc*2, ngc, 4, 2, 1, bias = False), #ngc, 32, 32
            nn.BatchNorm2d(ngc),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngc, 3, 4, 2, 1, bias = False), #3, 64, 64
            nn.Hardtanh(0,1),
        	)

    def forward(self, x):
    	return self.dec(x)

class lightingModel(nn.Module):
	def __init__(self, ngc=64, leaky_relu_param=0.05, zdim=1024):
		super(lightingModel, self).__init__()
		self.E = Encoder(ngc, leaky_relu_param, zdim)
		self.D = Decoder(ngc, zdim)

	def forward(self, x):
		lightCode, nonLightCode = self.E(x)
	        #print("LightCode shape =  ", lightCode.size())		
		z = torch.cat([lightCode, nonLightCode],1)
		#print("z shape before = " + str(z.size()))
	        z = z.unsqueeze(2)
	        z = z.unsqueeze(3)
	        #print("z shape = " + str(z.size()))
		o = self.D(z)
		return lightCode, nonLightCode, o


def cosine_embedding_loss(x1, x2):
	return nn.CosineEmbeddingLoss(x1, x2, margin=0.5)

def bce_loss(input, target):
	return nn.BCELoss(input, target)

class Loss(nn.Module):
	def __init__(self, margin=0.5):
		super(Loss, self).__init__()
		self.margin = margin

	def forward(self, outputs):
		#import math	
		"""oss = torch.max(F.pairwise_distance(outputs[0][0], outputs[1][0]))
		print("Loss shape = " + str(loss.size(0)))
		print("Loss value = " + str(loss.data))
		
		loss = torch.max(nn.PairwiseDistance(outputs[0][0], outputs[1][0]) -nn.PairwiseDistance(outputs[0][0], outputs[2][0]) + self.margin, 0) \
								+ torch.max(nn.PairwiseDistance(outputs[0][1], outputs[2][1]) -nn.PairwiseDistance(outputs[0][1], outputs[1][1]) + self.margin, 0) \
								+ nn.BCELoss(outputs[0][2], outputs[0][3]) + nn.BCELoss(outputs[1][2], outputs[1][3]) + nn.BCELoss(outputs[2][2], outputs[2][3])
		"""
		loss_func = nn.MSELoss()
		# cos = nn.CosineSimilarity(dim=1, eps=1e-6)
		cos = nn.CosineEmbeddingLoss(margin=0.3)
		# loss = torch.mean(cos(outputs[0][0], outputs[1][0]) - cos(outputs[0][0], outputs[2][0])) + torch.mean(cos(outputs[0][1], outputs[2][1]) - cos(outputs[0][1], outputs[1][1])) + loss_func(outputs[0][2], outputs[0][3]) + loss_func(outputs[1][2], outputs[1][3]) + loss_func(outputs[2][2], outputs[2][3])
		N,L = outputs[0][0].size()
		# print(outputs[0][0].size())
		Z = Variable(torch.zeros(N).cuda())
		# loss = torch.sum(torch.max(cos(outputs[0][0], outputs[1][0]) - cos(outputs[0][0], outputs[2][0]), Z)) + torch.sum(torch.max(cos(outputs[0][1], outputs[2][1]) - cos(outputs[0][1], outputs[1][1]), Z)) + loss_func(outputs[0][2], outputs[0][3]) + loss_func(outputs[1][2], outputs[1][3]) + loss_func(outputs[2][2], outputs[2][3]) 
		# loss = torch.sum(torch.max(torch.abs(cos(outputs[0][0], outputs[1][0])) - torch.abs(cos(outputs[0][0], outputs[2][0])) + self.margin, Z)) + torch.sum(torch.max(torch.abs(cos(outputs[0][1], outputs[2][1])) - torch.abs(cos(outputs[0][1], outputs[1][1])) + self.margin, Z)) + loss_func(outputs[0][2], outputs[0][3]) + loss_func(outputs[1][2], outputs[1][3]) + loss_func(outputs[2][2], outputs[2][3])
		loss = cos(outputs[0][0], outputs[1][0], Variable(torch.ones(N).cuda())) + cos(outputs[0][1], outputs[1][1], Variable(-1*torch.ones(N).cuda())) + cos(outputs[0][1], outputs[2][1], Variable(torch.ones(N).cuda())) + cos(outputs[0][0], outputs[2][0], Variable(-1*torch.ones(N).cuda())) + loss_func(outputs[0][2], outputs[0][3])
		"""
		loss = cos(outputs[0][0], outputs[1][0]) - cos(outputs[0][0], outputs[2][0]) + cos(outputs[0][1], outputs[2][1]) - cos(outputs[0][1], outputs[1][1]) + nn.BCELoss(outputs[0][2], outputs[0][3]) + nn.BCELoss(outputs[1][2], outputs[1][3]) + nn.BCELoss(outputs[2][2], outputs[2][3]) """
		return loss


class Trainer(object):
	def __init__(self, ae, params, optimizer="Adam"):
		self.ae = ae
		# self.latent_loss = loss_function[0]
		# self.reconstruction_loss = loss_function[1]
		self.loss_criterion = Loss()
		if optimizer=="Adam":
			self.optimizer = torch.optim.Adam(self.ae.parameters(), params.lr, [params.beta1, params.beta2])

	def ae_step(self, input1, input2, input3): #input1 = 0, input2 = 1, input3 = 9
		lightCode2, nonLightCode2, o2 = self.ae(input2)
		lightCode3, nonLightCode3, o3 = self.ae(input3)
		# assert torch.sum(torch.abs(o2-o3))[0].cpu() > 0
		self.optimizer.zero_grad()
		lightCode1, nonLightCode1, o1 = self.ae(input1)
		outputs = []
		outputs.append([lightCode1, nonLightCode1, o1, input1])
		outputs.append([lightCode2, nonLightCode2, o2, input2])
		outputs.append([lightCode3, nonLightCode3, o3, input3])
		# self.optimizer.zero_grad()
		loss = self.loss_criterion(outputs)
		loss.backward()
		self.optimizer.step()
		return loss		



model = lightingModel()
print(model.parameters())



