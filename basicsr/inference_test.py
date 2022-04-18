# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import sys
sys.path.append("/content/drive/MyDrive/DERAIN/HINet")
import importlib
import logging
from unittest import TestLoader
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str
from basicsr.utils import get_root_logger, imwrite, tensor2img
from tqdm import tqdm
from copy import deepcopy
import time, cv2
from skimage.measure import compare_psnr, compare_ssim
import numpy as np


# ----------------- from TransWeather ------------------
def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_psnr(im1_y, im2_y)]
    return ans

def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_ssim(im1_y, im2_y)]
    return ans
# ------------------------------------------------------


def inference(model, dataloader, opt, current_iter, 
                        save_img=True, rgb2bgr=True, use_image=True):
    metric_module = importlib.import_module('basicsr.metrics')
    dataset_name = dataloader.dataset.opt['name']
    with_metrics = opt['val'].get('metrics') is not None
    if with_metrics:
        metric_results = {
            metric: 0
            for metric in opt['val']['metrics'].keys()
        }
    pbar = tqdm(total=len(dataloader), unit='image')

    all_inference_time = []
    psnr_list = []
    ssim_list = []

    cnt = 0

    for idx, val_data in enumerate(dataloader):
        img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
        start_time = time.time()
        model.feed_data(val_data)
        if opt['val'].get('grids', False):
            model.grids()

        model.test()

        if opt['val'].get('grids', False):
            model.grids_inverse()

        visuals = model.get_current_visuals()

        all_inference_time.append(time.time() - start_time)

        sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
        gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
        del model.gt

        del model.lq
        del model.output
        torch.cuda.empty_cache()

        if save_img:
            if opt['is_train']:
                save_img_path = osp.join(opt['path']['visualization'],
                                            img_name,
                                            f'{img_name}_{current_iter}.png')
            else:
                save_img_path = osp.join(
                    opt['path']['visualization'], dataset_name,
                    f'{img_name}.png')
                
            imwrite(sr_img, save_img_path)
        
        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(sr_img, gt_img))
        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(sr_img, gt_img))


        pbar.update(1)
        pbar.set_description(f'Test {img_name}')
        cnt += 1
    pbar.close()

    current_metric = 0.
    
    avr_psnr = sum(psnr_list) / (len(psnr_list) + 1e-10)
    avr_ssim = sum(ssim_list) / (len(ssim_list) + 1e-10)
    print("[RESULTS] PSNR: {:.4f}, SSIM: {:.4f}, Average time: {:.4f} ms".format(avr_psnr, avr_ssim, np.mean(all_inference_time)*1000))

    return current_metric



def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    testset_root_dir = opt['datasets']['test']['dataroot_gt']
    subdirs = ['dawn_cloudy', 'night_outdoors', 'sunny_outdoors', 'underground']
    input_subs = ['rain_L', 'rain_H']

    # create model
    model = create_model(opt)

    for subdir in subdirs:
        for input_sub in input_subs:
            print("=====> Currently running: ", subdir, " with ", input_sub)
            opt['datasets']['test']['dataroot_gt'] = osp.join(testset_root_dir, subdir, 'gt')
            opt['datasets']['test']['dataroot_lq'] = osp.join(testset_root_dir, subdir, input_sub)

            # create test dataset and dataloader
            test_loaders = []
            for phase, dataset_opt in sorted(opt['datasets'].items()):
                print("========> ", dataset_opt)
                test_set = create_dataset(dataset_opt)
                test_loader = create_dataloader(
                    test_set,
                    dataset_opt,
                    num_gpu=opt['num_gpu'],
                    dist=opt['dist'],
                    sampler=None,
                    seed=opt['manual_seed'])
                logger.info(
                    f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
                test_loaders.append(test_loader)

            for test_loader in test_loaders:
                test_set_name = test_loader.dataset.opt['name']
                logger.info(f'Testing {test_set_name}...')
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)

                inference(model, test_loader, opt, opt['name'],
                                save_img=True, rgb2bgr=True, use_image=True)


if __name__ == '__main__':
    main()
