import os
import parameters
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.image

from src.logger import create_logger
from src.loader import load_images, DataSampler
from src.utils import bool_flag

# params = parameters.interpolateParams()
# # assert os.path.isfile(params.model_path)
# assert params.n_images >= 1 and params.n_interpolations >= 2
# logger = create_logger(None)
# # ae = torch.load(params.model_path).eval()
# # params.debug = True
# # params.batch_size = 32
# # params.v_flip = False
# # params.h_flip = False
# # params.img_sz = ae.img_sz
# # params.attr = ae.attr
# # params.n_attr = ae.n_attr
# # if not (len(params.attr) == 1 and params.n_attr == 2):
# #     raise Exception("The model must use a single boolean attribute only.")
# data, attributes = load_images(params)
# test_data = DataSampler(data[2], attributes[2], params)




def interpolate(ae,n_epoch):

    def get_interpolations(ae, images, attributes, params):
        """
        Reconstruct images / create interpolations
        """
        assert len(images) == len(attributes)
        enc_outputs = ae.encode(images)

        # interpolation values
        alphas = np.linspace(1 - params.alpha_min, params.alpha_max, params.n_interpolations)
        alphas = [torch.FloatTensor([1 - alpha, alpha]) for alpha in alphas]

        # original image / reconstructed image / interpolations
        outputs = []
        outputs.append(images)
        outputs.append(ae.decode(enc_outputs, attributes)[-1])
        for alpha in alphas:
            alpha = Variable(alpha.unsqueeze(0).expand((len(images), 2)).cuda())
            outputs.append(ae.decode(enc_outputs, alpha)[-1])

        # return stacked images
        return torch.cat([x.unsqueeze(1) for x in outputs], 1).data.cpu()
    params = parameters.interpolateParams()
    # ae = torch.load(params.model_path).eval()
    params.debug = True
    params.batch_size = 50
    params.v_flip = False
    params.h_flip = False
    params.img_sz = ae.img_sz
    params.attr = ae.attr
    params.n_attr = ae.n_attr
    if not (len(params.attr) == 1 and params.n_attr == 2):
        raise Exception("The model must use a single boolean attribute only.")
    data, attributes = load_images(params)
    test_data = DataSampler(data[2], attributes[2], params)

    interpolations = []

    for k in range(0, params.n_images, 100):
        i = params.offset + k
        j = params.offset + min(params.n_images, k + 100)
        images, attributes = test_data.eval_batch(i, j)
        interpolations.append(get_interpolations(ae, images, attributes, params))

        interpolations = torch.cat(interpolations, 0)
        assert interpolations.size() == (params.n_images, 2 + params.n_interpolations,
                                         3, params.img_sz, params.img_sz)


    def get_grid(images, row_wise, plot_size=5):
        """
        Create a grid with all images.
        """
        n_images, n_columns, img_fm, img_sz, _ = images.size()
        if not row_wise:
            images = images.transpose(0, 1).contiguous()
        images = images.view(n_images * n_columns, img_fm, img_sz, img_sz)
        images.add_(1).div_(2.0)
        return make_grid(images, nrow=(n_columns if row_wise else n_images))


    # generate the grid / save it to a PNG file
    grid = get_grid(interpolations, params.row_wise, params.plot_size)
    matplotlib.image.imsave(params.output_path+str(n_epoch)+".png", grid.numpy().transpose((1, 2, 0)))

