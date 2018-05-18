import os
import torch
import parameters


from src.loader import load_images, DataSampler
from src.utils import initialize_exp, bool_flag, attr_flag, check_attr
from src.model import AutoEncoder, LatentDiscriminator
from src.training import Trainer
from interpolate import *
# from src.evaluation import Evaluator

params = parameters.trainParams()
params.attr = attr_flag(params.attr)
check_attr(params)
print("params checked")
logger = initialize_exp(params)
data, attributes = load_images(params)
train_data = DataSampler(data[0], attributes[0], params)
valid_data = DataSampler(data[1], attributes[1], params)
ae = AutoEncoder(params).cuda()
lat_dis = LatentDiscriminator(params).cuda() if params.n_lat_dis else None
print("data and model loaded and created")

trainer = Trainer(ae, lat_dis, None, None, train_data, params)
# evaluator = Evaluator(ae, lat_dis, None, None, None, valid_data, params)
print("Trainer created. Starting Training")
for n_epoch in range(params.n_epochs):
    logger.info('Starting epoch %i...' % n_epoch)
    for n_iter in range(0, params.epoch_size, params.batch_size):
    	print("n_iter = " + str(n_iter))
	trainer.lat_dis_step()
        trainer.autoencoder_step()
        trainer.step(n_iter)
    logger.info('End of epoch %i.\n' % n_epoch)
    if n_epoch%5 == 0:
	interpolate(trainer.ae, n_epoch)
print("Training End. Saving Model!")
trainer.save_model('final')

