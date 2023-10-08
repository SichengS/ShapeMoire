import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.loss_util import *
from utils.common import *
from torch.nn.parameter import Parameter
from functools import partial
import time

def model_fn_decorator(loss_fn, device, mode='train'):
    def test_model_fn(args, data, model, save_path, compute_metrics):
        # prepare input and forward
        number = data['number']
        cur_psnr = 0.0
        cur_ssim = 0.0
        cur_lpips = 0.0

        in_img = data['in_img'].to(device)
        label = data['label'].to(device)
        b, c, h, w = in_img.size()

        # pad image such that the resolution is a multiple of 32
        w_pad = (math.ceil(w/32)*32 - w) // 2
        h_pad = (math.ceil(h/32)*32 - h) // 2
        w_odd_pad = w_pad
        h_odd_pad = h_pad
        if w % 2 == 1:
            w_odd_pad += 1
        if h % 2 == 1:
            h_odd_pad += 1

        in_img = img_pad(in_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)

        with torch.no_grad():
            st = time.time()

            mid_1, mid_2, mid_3, out_1, out_2, out_3 = model(in_img)

            cur_time = time.time()-st
            if h_pad != 0:
                mid_1 = mid_1[:, :, h_pad:-h_odd_pad, :]
                out_1 = out_1[:, :, h_pad:-h_odd_pad, :]
            if w_pad != 0:
                mid_1 = mid_1[:, :, :, w_pad:-w_odd_pad]
                out_1 = out_1[:, :, :, w_pad:-w_odd_pad]

        if args.EVALUATION_METRIC:
            cur_lpips_mid, cur_psnr_mid, cur_ssim_mid = compute_metrics.compute(mid_1, label)
            cur_lpips_out, cur_psnr_out, cur_ssim_out = compute_metrics.compute(mid_1 + out_1, label)

        # save images
        if args.SAVE_IMG:
            out_save = mid_1.detach().cpu()
            torchvision.utils.save_image(out_save, save_path + '/' + 'test_%s' % number[0] + '.%s' % args.SAVE_IMG)

        return cur_lpips_mid, cur_psnr_mid, cur_ssim_mid, cur_lpips_out, cur_psnr_out, cur_ssim_out, cur_time
    
    def validate_model_fn(args, data, model, compute_metrics):
        # prepare input and forward
        number = data['number']
        cur_psnr = 0.0

        in_img = data['in_img'].to(device)
        label = data['label'].to(device)
        b, c, h, w = in_img.size()

        # pad image such that the resolution is a multiple of 32
        w_pad = (math.ceil(w/32)*32 - w) // 2
        h_pad = (math.ceil(h/32)*32 - h) // 2
        w_odd_pad = w_pad
        h_odd_pad = h_pad
        if w % 2 == 1:
            w_odd_pad += 1
        if h % 2 == 1:
            h_odd_pad += 1
        in_img = img_pad(in_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
        
        with torch.no_grad():
            st = time.time()
            out_1, out_2, out_3 = model(in_img)
           
            cur_time = time.time()-st
            if h_pad != 0:
               out_1 = out_1[:, :, h_pad:-h_odd_pad, :]
            if w_pad != 0:
               out_1 = out_1[:, :, :, w_pad:-w_odd_pad]
        
        if args.EVALUATION_METRIC:
            cur_psnr = compute_metrics.compute(out_1, label)

        return cur_psnr

    def model_fn(args, data, model, iters):
        model.train()
        # prepare input and forward
        in_img = data['in_img'].to(device)
        base_in_img = torch.mean(in_img, dim = [2,3], keepdim = True)
        shape_in_img = in_img - base_in_img
        in_img = torch.cat((in_img, shape_in_img), dim=0)

        label = data['label'].to(device)
        base_label = torch.mean(label, dim = [2,3], keepdim = True)
        shape_label = label - base_label
        label = torch.cat((label, shape_label), dim=0)
        
        out_1, out_2, out_3 = model(in_img)
        loss = loss_fn(out_1, out_2, out_3, label)

        return loss

    if mode == 'test':
        return test_model_fn
    else:
        return model_fn, validate_model_fn