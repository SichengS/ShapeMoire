import os
import numpy as np
import time
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models.vgg as vgg
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from config.config import args
from tqdm import tqdm
from load_data import * 
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.metric_train import create_metrics

if args.TEST_BASELINE:
    from model_dense import *
else:
    from model_shapeconv import *
    args.EXP_NAME = 'ShapeMoire'

def compute_l1_loss(input, output):
    return torch.mean(torch.abs(input-output))
        
def loss_Textures(x, y, nc=3, alpha=1.2, margin=0):
  xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
  yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))
  
  xi2 = torch.sum(xi * xi, dim=2)
  yi2 = torch.sum(yi * yi, dim=2)
  #pdb.set_trace()    #15*32*32
  out = nn.functional.relu(yi2.mul(alpha) - xi2 + margin)
  
  return torch.mean(out)

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
        
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
        

save_path = 'saved_models' + '/%s' %(args.DATA_TYPE)+ '/%s' %(args.EXP_NAME) + f'/net_checkpoints'
os.makedirs(save_path, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# random seed
random.seed(args.SEED)
np.random.seed(args.SEED)
torch.manual_seed(args.SEED)
torch.cuda.manual_seed_all(args.SEED)
if args.SEED == 0:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()#  smoothl1loss()
tvloss = TVLoss()
lossmse = torch.nn.MSELoss()


# if use GAN loss
lambda_pixel = 100
patch = (1, args.img_height//2**4, args.img_width//2**4)   

# Initialize wdnet
generator = WDNet()


wavelet_dec = WaveletTransform(scale=2, dec=True)
wavelet_rec = WaveletTransform(scale=2, dec=False)          

if cuda:
    generator = generator.cuda()
    
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    lossnet = LossNetwork().float().cuda()
    wavelet_dec = wavelet_dec.cuda()
    wavelet_rec = wavelet_rec.cuda()

if args.LOAD_EPOCH != 0:
    generator = generator.load_state_dict(torch.load('./saved_models/facades2/lastest.pth' ))#%  args.epoch))
   

else:
    # Initialize weights
    generator.apply(weights_init_normal)
    
device = torch.device("cuda:0")


# argsimizers
argsimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(args.b1, args.b2))

            
mytransform = transforms.Compose([    
     transforms.ToTensor(),])
    
# change the root to your own data path
TrainImgLoader = create_dataset(args, data_path=args.TRAIN_DATASET, mode='train')
print('data loader finish！')


def get_mask(dg_img,img):
    mask = np.fabs(dg_img.cpu()-img.cpu())
    mask[mask<(20.0/255.0)] = 0.0
    mask = mask.cuda()
    return mask

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_images(epoch , i ,real_A,real_B,fake_B):
    data,pred,label = real_A *255 , fake_B *255, real_B *255
    data = data.cpu()
    pred = pred.cpu()
    label = label.cpu()
    #pdb.set_trace()
    pred = torch.clamp(pred.detach(),0,255)
    data,pred,label = data.int(),pred.int(),label.int()
    h,w = pred.shape[-2],pred.shape[-1]
    img = np.zeros((h,1*3*w,3))
    #pdb.set_trace()
    for idx in range(0,1):
        row = idx*h
        tmplist = [data[idx],pred[idx],label[idx]]
        for k in range(3):
            col = k*w
            tmp = np.transpose(tmplist[k],(1,2,0))
            img[row:row+h,col:col+w]=np.array(tmp)
    #pdb.set_trace()
    img = img.astype(np.uint8)
    img= Image.fromarray(img)
    img.save("./train_result/%03d_%06d.png"%(epoch,i))
    
# ----------
#  Training
# ----------
EPS = 1e-12
prev_time = time.time()
step = 0
best_psnr=0
for epoch in range(args.LOAD_EPOCH+1, args.EPOCHS+1):
    tbar = tqdm(TrainImgLoader)
    for batch_idx, data in enumerate(tbar):
        step = step+1
        
        # set lr rate
        current_lr = 0.0002*(1/2)**(step/100000)
        for param_group in argsimizer_G.param_groups:
            param_group["lr"] = current_lr
            
        # Model inputs
        img_train = data['in_img'].cuda()
        label = data['label'].cuda()

        
        #ShapeMoire
        if not args.TEST_BASELINE:
            base_img_train = torch.mean(img_train, dim = [2,3], keepdim = True)
            shape_img_train = img_train - base_img_train
            img_train = torch.cat((img_train, shape_img_train), dim=0)
            base_label = torch.mean(label, dim = [2,3], keepdim = True)
            shape_label = label - base_label
            label = torch.cat((label, shape_label), dim=0)
        

        real_A, real_B = Variable(img_train), Variable(label)
        #pdb.set_trace() 
        x_r = (real_A[:,0,:,:]*255-105.648186)/255.+0.5
        x_g = (real_A[:,1,:,:]*255-95.4836)/255.+0.5
        x_b = (real_A[:,2,:,:]*255-86.4105)/255.+0.5
        real_A = torch.cat([ x_r.unsqueeze(1) ,x_g.unsqueeze(1) ,x_b.unsqueeze(1)  ],1)
  
        y_r = ((real_A[:,0,:,:]-0.5)*255+121.2556)/255.
        y_g = ((real_A[:,1,:,:]-0.5)*255+114.89969)/255.
        y_b = ((real_A[:,2,:,:]-0.5)*255+102.02478)/255.
        real_A = torch.cat([ y_r.unsqueeze(1) , y_g.unsqueeze(1) , y_b.unsqueeze(1)  ],1)
        
        #121.2556, 114.89969, 102.02478
        target_wavelets = wavelet_dec(real_B)
        batch_size = real_B.size(0)
        wavelets_lr_b = target_wavelets[:,0:3,:,:]
        wavelets_sr_b = target_wavelets[:,3:,:,:]
        
        source_wavelets = wavelet_dec(real_A)
        
        argsimizer_G.zero_grad()

        
        tensor_c = torch.from_numpy(np.array([123.6800, 116.7790, 103.9390]).astype(np.float32).reshape((1,3,1,1))).cuda() 
            
        wavelets_fake_B_re = generator(source_wavelets)
            
        fake_B = wavelet_rec(wavelets_fake_B_re) +  real_A       
            
        wavelets_fake_B    = wavelet_dec(fake_B)
        wavelets_lr_fake_B = wavelets_fake_B[:,0:3,:,:]
        wavelets_sr_fake_B = wavelets_fake_B[:,3:,:,:]
            
        loss_GAN = 0.0
       
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)   #.................................

        # preceptual loss
        loss_fake_B = lossnet(fake_B*255-tensor_c)
        loss_real_B = lossnet(real_B*255-tensor_c)
        p0=compute_l1_loss(fake_B*255-tensor_c,real_B*255-tensor_c)*2
        p1=compute_l1_loss(loss_fake_B['relu1'],loss_real_B['relu1'])/2.6
        p2=compute_l1_loss(loss_fake_B['relu2'],loss_real_B['relu2'])/4.8

        loss_p = p0+p1+p2   #+p3+p4+p5
            
           
        loss_lr = compute_l1_loss(wavelets_lr_fake_B[:,0:3,:,:],  wavelets_lr_b )
        loss_sr = compute_l1_loss(wavelets_sr_fake_B,  wavelets_sr_b )
        loss_textures = loss_Textures(wavelets_sr_fake_B, wavelets_sr_b)

        loss_G = 0.001*loss_GAN + (  1*loss_p) + loss_sr.mul(100) + loss_lr.mul(10) + loss_textures.mul(5)  # +  loss_tv  loss_pixel    
        loss_G.backward()
    
        argsimizer_G.step()        

        desc = 'Training  : Epoch %d, lr %.7f, Avg. Loss = %.5f' % (epoch, current_lr, loss_G)
        tbar.set_description(desc)
        tbar.update()
        
    # validate every epoch after total epoch/2
    if epoch >= 1:#args.EPOCHS/2:
        args.BATCH_SIZE = 1
        TestImgLoader = create_dataset(args, data_path=args.TEST_DATASET, mode='test')
        compute_metrics = create_metrics(args, device=device)
        tbar = tqdm(TestImgLoader)

        for batch_idx, data in enumerate(tbar):
            img = data['in_img'].to(device)
            imggt = data['label'].to(device)

            img2 = predict_img(net=generator, img=img, lossnet=lossnet)
            cur_psnr = compute_metrics.compute(img2, imggt)        
        
        # save best epoch
        bestfilename = save_path + f'/best_epoch{epoch}_{cur_psnr:.4f}.pth'
        if best_psnr <= cur_psnr:
            torch.save(generator.state_dict(), bestfilename)

            best_psnr = cur_psnr
    
      
