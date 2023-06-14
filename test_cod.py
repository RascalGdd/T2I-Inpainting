import os
import sys
import json
import numpy as np
import cv2
import torch
import random
from PIL import Image
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast
from torch.utils.data import Dataset
from lavis.models import load_model_and_preprocess
from ldm.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)

torch.set_grad_enabled(False)



def blip_color(raw_img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True,
                                                                      device=device)
    question = "The background color is {}"
    image = vis_processors["eval"](raw_img).unsqueeze(0).to(device)
    question = txt_processors["eval"](question)
    prompt = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
    return prompt[0]



def main():
    supported_cond = [e.name for e in ExtraCondition]
    parser = get_base_argument_parser()
    parser.add_argument(
        '--which_cond',
        type=str,
        required=True,
        choices=supported_cond,
        help='which condition modality you want to test',
    )


    opt = parser.parse_args()   
    os.makedirs(opt.outdir, exist_ok=True)   
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    which_cond = opt.which_cond
    if opt.resize_short_edge is None:
        print(f"you don't specify the resize_shot_edge, so the maximum resolution is set to {opt.max_resolution}")

    #############################
    folder_cond = '/cluster/scratch/denfan/inpainting_stable/cond/'
    folder_img = '/cluster/scratch/denfan/inpainting_stable/background1/'
    folder_mask = '/cluster/work/cvl/denfan/diandian/control/inpainting/datasets/camo_diff_512/camo_mask/'
    
    for filename in os.listdir(folder_img):
          
        # makedir
        name = filename.split('.')
        name = name[-2] 
        #if not os.path.isdir(os.path.join('/cluster/scratch/denfan/inpainting_stable/rand_seed/',name)):         
            #if not os.path.isdir(os.path.join('/cluster/scratch/denfan/inpainting_stable/rand_seed/dd/',name)):
        if (not os.path.isdir(os.path.join('/cluster/scratch/denfan/inpainting_stable/rand_seed/', name))) and (not os.path.isdir(os.path.join('/cluster/scratch/denfan/inpainting_stable/rand_seed/dd/', name))):
            os.makedirs(os.path.join(opt.outdir,name)) 
       
            path_img = os.path.join(folder_img, filename)
            path_mask = os.path.join(folder_mask, filename).replace('jpg', 'png')
            cond_path = os.path.join(folder_cond, filename)
           
            parts = filename.split('-')
            prompt = parts[-2]
            print(prompt)
            # ---------  add blip ---------
            raw_img = Image.open(os.path.join(folder_cond, filename))
            color = blip_color(raw_img)
            prompt = 'a ' + color + ' '+prompt + ', best quality, photorealistic and realistic'
            print(prompt)
    
    
            sd_model, sampler = get_sd_models(opt)
            adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))
            cond_model = None
            #if opt.cond_inp_type == 'image':
                #cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond))
            cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond))    
            process_cond_module = getattr(api, f'get_cond_{which_cond}')
     
            # colormap
            cond = process_cond_module(opt, cond_path, opt.cond_inp_type, cond_model)
            adapter_features, append_to_context = get_adapter_feature(cond, adapter)
            cv2.imwrite(os.path.join('/cluster/scratch/denfan/inpainting_stable/rand_seed/', 'color',filename.replace('jpg','png')), tensor2img(cond))  
            
            for index in np.arange(100):        
                # inference
                with torch.inference_mode(), \
                        sd_model.ema_scope(), \
                        autocast('cuda'):
                    seed=random.randint(0, 10000)
                    seed_everything(seed)
                    for v_idx in range(opt.n_samples):
                        # result
                        result = diffusion_inference(path_img,path_mask,prompt,opt, sd_model, sampler, adapter_features, append_to_context)
                        result.save(os.path.join(opt.outdir,name, name+'-seed'+str(seed)+'.jpg'))

        else:
            continue

    
if __name__ == '__main__':
    main()










