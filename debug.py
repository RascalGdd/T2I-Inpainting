import argparse
import logging
import os
import os.path as osp
import time
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           img2tensor, scandir, tensor2img)
from basicsr.utils.options import copy_opt_file, dict2str
from omegaconf import OmegaConf
from PIL import Image
from ldm.data.dataset_coco import dataset_cod_mask
from dist_util import get_bare_model, get_dist_info, init_dist, master_only
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.adapter import Adapter_light
from ldm.util import instantiate_from_config
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)
from ldm.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)
from ldm.util import fix_cond_shapes, load_model_from_config, read_state_dict
from pytorch_lightning import seed_everything


@master_only
def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(experiments_root, 'models'))
    os.makedirs(osp.join(experiments_root, 'training_states'))
    os.makedirs(osp.join(experiments_root, 'visualization'))

def load_resume_state(opt):
    resume_state_path = None
    if opt.auto_resume:
        state_path = osp.join('experiments', opt.name, 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt.resume_state_path = resume_state_path
    # else:
    #     if opt['path'].get('resume_state'):
    #         resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        # check_resume(opt, resume_state['iter'])
    return resume_state

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bsize",
    type=int,
    default=1,
    help="the prompt to render"
)
parser.add_argument(
    "--epochs",
    type=int,
    #default=10000,
    default=500,
    help="the prompt to render"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=1,
    help="the prompt to render"
)
parser.add_argument(
    "--use_shuffle",
    type=bool,
    default=True,
    help="the prompt to render"
)
parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
)
parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
)
parser.add_argument(
        "--auto_resume",
        action='store_true',
        help="use plms sampling",
)
parser.add_argument(
        "--ckpt",
        type=str,
        #default="models/sd-v1-4.ckpt",
        default="models/sd-v1-5-inpainting.ckpt",
        help="path to checkpoint of model",
)
parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/sd-v1-inference.yaml",
        help="path to config which constructs model",
)
parser.add_argument(
        "--print_fq",
        type=int,
        default=100,
        help="path to config which constructs model",
)
parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
)
parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
        "--scale",
        type=float,
        #default=7.5,
        default=9,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
        "--gpus",
        default=[0,1,2,3],
        help="gpu idx",
)
parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        help='node rank for distributed training'
)
parser.add_argument(
        '--launcher',
        default='pytorch',
        type=str,
        help='node rank for distributed training'
)
parser.add_argument(
        '--l_cond',
        default=4,
        type=int,
        help='number of scales'
)
opt = parser.parse_args()
opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
opt.vae_ckpt = None

opt.cond_tau=1.0
opt.style_cond_tau=1.0

if __name__ == '__main__':
    config = OmegaConf.load(f"{opt.config}")
    opt.name = 'train_color_one'
    #opt.name = 'train_color'
    #opt.name = 'train_color_latent'
    
    os.environ['RANK']='0'
    os.environ['WORLD_SIZE']='1'
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']='1234'

    # distributed setting
    init_dist(opt.launcher)
    torch.backends.cudnn.benchmark = True
    device='cuda'
    torch.cuda.set_device(opt.local_rank)

    # dataset
    path_json_train = '/cluster/work/cvl/denfan/diandian/control/inpainting/datasets/camo_diff_512/one.json'
    path_json_val = '/cluster/work/cvl/denfan/diandian/control/inpainting/datasets/camo_diff_512/short.json'
    
    train_dataset = dataset_cod_mask(path_json_train,
    root_path_im='/cluster/work/cvl/denfan/diandian/control/inpainting/datasets/camo_diff_512/camo_target',
    root_path_mask='/cluster/work/cvl/denfan/diandian/control/inpainting/datasets/camo_diff_512/camo_mask',
    image_size=512
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_dataset = dataset_cod_mask(path_json_val,
    root_path_im='/cluster/work/cvl/denfan/diandian/control/inpainting/datasets/camo_diff_512/camo_target',
    root_path_mask='/cluster/work/cvl/denfan/diandian/control/inpainting/datasets/camo_diff_512/camo_mask',
    image_size=512
    )
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.bsize,
            shuffle=(train_sampler is None),
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False)

    
    
    # inpainting
    #model = load_model_from_config(config, f"{opt.ckpt}").to(device)
    model = load_model_from_config(config, opt.ckpt, opt.vae_ckpt).to(opt.device)

    # t2i encoder
    model_ad = Adapter_light(channels=[320, 640, 1280, 1280][:4],cin=192, nums_rb=4).to(device)    
    

    
    adapter = {}
    adapter['model'] = model_ad
    adapter['cond_weight'] = 1.0

    # to gpus
    model_ad = torch.nn.parallel.DistributedDataParallel(
        model_ad,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)

    # optimizer
    params = list(model_ad.parameters())
    optimizer = torch.optim.AdamW(params, lr=config['training']['lr'])

    experiments_root = osp.join('experiments', opt.name)

    # resume state
    resume_state = load_resume_state(opt)
    if resume_state is None:
        mkdir_and_rename(experiments_root)
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
    else:
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
        resume_optimizers = resume_state['optimizers']
        optimizer.load_state_dict(resume_optimizers)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)
    neg_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
          'fewer digits, cropped, worst quality, low quality'    

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    seed_list = list()

    for epoch in range(start_epoch, opt.epochs):
        # seed
        seed = random.randint(0, 10000)
        seed_everything(seed)
        seed_list.append(seed)
        train_dataloader.sampler.set_epoch(epoch)
        # train
        for _, data in enumerate(train_dataloader):
            current_iter += 1
            with torch.no_grad():
                # filename
                filename = data['name']
                # source_img
                asd
  
                z = model.module.encode_first_stage((data['im']).cuda(non_blocking=True))
                z = model.module.get_first_stage_encoding(z)
                # mask
                c_cat = list()
                mask = data['mask'].to(device)
                bchw = [1, 4, 64, 64]
                mask = torch.nn.functional.interpolate(mask, size=bchw[-2:])
                c_cat.append(mask)
                # masked_img
                masked_img = data['masked_img'].to(device)
                masked_img = model.module.encode_first_stage(masked_img)
                masked_img = model.module.get_first_stage_encoding(masked_img)
                c_cat.append(masked_img)
                c_cat = [cc.to(device) for cc in c_cat]
                c_cat = torch.cat(c_cat, dim=1)
                # txt
                c = model.module.cond_stage_model.encode(data['sentence'])
                uc = model.module.get_learned_conditioning([neg_prompt])
                c, uc = fix_cond_shapes(model.module, c, uc)
                cond = {"c_concat": [c_cat], "c_crossattn": [c]}
                uc = {"c_concat": [c_cat], "c_crossattn": [uc]}