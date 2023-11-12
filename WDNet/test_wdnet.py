import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import torchvision.models.vgg as vgg
from model_dense import *
import pdb
from torchvision import transforms
from skimage import measure
from skimage import color
from config.config import args
from load_data import * 
from tqdm import tqdm
from utils.metric import create_metrics
import torchvision
criterion_GAN = torch.nn.MSELoss()
    
def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.double) - im2.astype(np.double)) ** 2).mean()
    psnr = 10 * np.log10(255.0 ** 2 / mse)
    return psnr   

class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '13': "relu3",
            '22': "relu4",
            '31': "relu5", 
        }
        
    def forward(self, x):
        output = {}

        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x

        return output
        
transform1 = transforms.Compose([
      transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
      #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
      ])       
def predict_img(net,
                #cls,
                img,
                lossnet,
                use_gpu=True):    
    net.eval()
    lossnet.eval()
    
    w,h = img.shape[1],img.shape[2]
    tensor_c = torch.from_numpy(np.array([123.6800, 116.7790, 103.9390]).astype(np.float32).reshape((1,3,1,1))).cuda()
    real_a_pre = lossnet((img*255-tensor_c)) 
    
    relu_1 = nn.functional.interpolate(real_a_pre['relu1'].detach(),size=(w,h))
    relu_2 = nn.functional.interpolate(real_a_pre['relu2'].detach(),size=(w,h))
    relu_3 = nn.functional.interpolate(real_a_pre['relu3'].detach(),size=(w,h))


    precept = torch.cat([relu_1/255.,relu_2/255.,relu_3/255.],1)#,relu_4/255.,relu_5/255.], 1)
    #img=img.unsqueeze(0)
    x_r = (img[:,0,:,:]*255-105.648186)/255.+0.5
    x_g = (img[:,1,:,:]*255-95.4836)/255.+0.5
    x_b = (img[:,2,:,:]*255-86.4105)/255.+0.5
    img = torch.cat([ x_r.unsqueeze(1) ,x_g.unsqueeze(1) ,x_b.unsqueeze(1)  ],1)
  
    y_r = ((img[:,0,:,:]-0.5)*255+121.2556)/255.
    y_g = ((img[:,1,:,:]-0.5)*255+114.89969)/255.
    y_b = ((img[:,2,:,:]-0.5)*255+102.02478)/255.
    img = torch.cat([ y_r.unsqueeze(1) , y_g.unsqueeze(1) , y_b.unsqueeze(1)  ],1)
    if use_gpu:
        img = img.cuda()
        net = net.cuda()
        
    with torch.no_grad():
        
        imgin = wavelet_dec(img)
        imgout = net(Variable(imgin))
        imgout =wavelet_rec(imgout) + img
    
    return imgout

crit = criterion_GAN.cuda() 

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()


if __name__ == "__main__":

    net = WDNet()
    lossnet= LossNetwork()
    wavelet_dec = WaveletTransform(scale=2, dec=True)
    wavelet_rec = WaveletTransform(scale=2, dec=False)        

    
    print("Begin Loading model {}".format(args.LOAD_PATH))

    if torch.cuda.is_available():
        print("Using CUDA version of the net, prepare your GPU !")
        device = "cuda"
        net.to(device)

        net.load_state_dict(torch.load(args.LOAD_PATH))
        lossnet.to(device)
        wavelet_dec.to(device)
        wavelet_rec.to(device)

    else:
        device = "cpu"
        net.to(device)
        net.load_state_dict(torch.load(args.LOAD_PATH, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    # create test dataset
    Test_path = args.TEST_DATASET
    args.BATCH_SIZE = 1
    TestImgLoader = create_dataset(args, data_path=Test_path, mode='test')
    compute_metrics = create_metrics(args, device=device)

    #log=open('result.txt','w')

    tbar = tqdm(TestImgLoader)

    total_lpips = 0
    total_psnr = 0
    total_ssim = 0

    for batch_idx, data in enumerate(tbar):

        img = data['in_img'].to(device)
        imggt = data['label'].to(device)

        img2 = predict_img(net=net, img=img, lossnet=lossnet)
        cur_lpips, cur_psnr, cur_ssim = compute_metrics.compute(img2, imggt)
        
        total_lpips += cur_lpips
        total_psnr += cur_psnr
        total_ssim += cur_ssim

        num = data['number'][0]
        #log.write('%s: LPIPS:%f, PSNR:%f, SSIM:%f' % (num, cur_lpips, cur_psnr, cur_ssim))

        out_save = img2.detach().cpu()
        save_path='saved_models' + '/%s' %(args.DATA_TYPE)+'/%s' %(args.EXP_NAME)+ '/test_result'

        os.makedirs(save_path, exist_ok=True)
        
        if args.SAVE_IMG:
            torchvision.utils.save_image(out_save, save_path + '/test_%s' % num + '.%s' % args.SAVE_IMG)
        
    avg_lpips=total_lpips/(batch_idx+1)
    avg_psnr=total_psnr/(batch_idx+1)
    avg_ssim=total_ssim/(batch_idx+1)
    print('avg_lpips:%f , avg_psnr:%f , avg_ssim:%f' % (avg_lpips, avg_psnr,avg_ssim))
    #log.close()
        
