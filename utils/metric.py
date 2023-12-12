from .common import SSIM, PSNR, tensor2img
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim
from utils.matlab_ssim import MATLAB_SSIM
import lpips
import torch
import numpy as np
from math import log10

class create_metrics():
    """
       We note that for different benchmarks, previous works calculate metrics in different ways, which might
       lead to inconsistent SSIM results (and slightly different PSNR), and thus we follow their individual
       ways to compute metrics on each individual dataset for fair comparisons.

    """
    def __init__(self, args, device, use_fast=False):
        self.data_type = args.DATA_TYPE
        self.use_fast = use_fast
        self.lpips_fn = lpips.LPIPS(net='alex').cuda()
        self.fast_ssim = SSIM()
        self.fast_psnr = PSNR()
        self.matlab_ssim = MATLAB_SSIM(device=device)

    def compute(self, out_img, gt):
        if self.use_fast:
            res_psnr, res_ssim = self.fast_psnr_ssim(out_img, gt)
        else:
            if self.data_type == 'UHDM':
                res_psnr, res_ssim = self.fast_psnr_ssim(out_img, gt)
            elif self.data_type == 'FHDMi':
                res_psnr, res_ssim = self.skimage_psnr_ssim(out_img, gt)
            elif self.data_type == 'TIP':
                res_psnr, res_ssim = self.matlab_psnr_ssim(out_img, gt)
            elif self.data_type == 'AIM':
                res_psnr, res_ssim = self.aim_psnr_ssim(out_img, gt)
            else:
                print('Unrecognized data_type for evaluation!')
                raise NotImplementedError
        pre = torch.clamp(out_img, min=0, max=1)
        tar = torch.clamp(gt, min=0, max=1)

        # calculate LPIPS
        res_lpips = self.lpips_fn.forward(pre, tar, normalize=True).item()
        return res_lpips, res_psnr, res_ssim


    def fast_psnr_ssim(self, out_img, gt):
        pre = torch.clamp(out_img, min=0, max=1)
        tar = torch.clamp(gt, min=0, max=1)
        psnr = self.fast_psnr(pre, tar)
        ssim = self.fast_ssim(pre, tar)
        return psnr, ssim

    def skimage_psnr_ssim(self, out_img, gt):

        mi1 = tensor2img(out_img)
        mt1 = tensor2img(gt)
        psnr = ski_psnr(mt1, mi1)
        ssim = ski_ssim(mt1, mi1, multichannel=True)
        return psnr, ssim

    def matlab_psnr_ssim(self, out_img, gt):

        mi1 = tensor2img(out_img)
        mt1 = tensor2img(gt)
        psnr = ski_psnr(mt1, mi1)
        ssim = self.matlab_ssim(mt1, mi1)
        return psnr, ssim

    def aim_psnr_ssim(self, out_img, gt):

        mi1 = tensor2img(out_img)
        mt1 = tensor2img(gt)
        mi1 = mi1.astype(np.float32) / 255.0
        mt1 = mt1.astype(np.float32) / 255.0
        psnr = 10 * log10(1 / np.mean((mt1 - mi1) ** 2))
        ssim = ski_ssim(mt1, mi1, multichannel=True)
        return psnr, ssim
        