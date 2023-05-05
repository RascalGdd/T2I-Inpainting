import os

import cv2
import torch
from PIL import Image
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast
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
    ### add blip ###
    raw_img = Image.open(opt.cond_path)
    color = blip_color(raw_img)
    parts = opt.prompt.split()
    opt.prompt = parts[0] + ' ' + color + ' ' + parts[1]
    opt.prompt = opt.prompt+', best quality, photorealistic and realistic'
    print(opt.prompt)
    ###

    which_cond = opt.which_cond
    if opt.outdir is None:
        opt.outdir = f'outputs/test-{which_cond}'
    os.makedirs(opt.outdir, exist_ok=True)
    if opt.resize_short_edge is None:
        print(f"you don't specify the resize_shot_edge, so the maximum resolution is set to {opt.max_resolution}")
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # support two test mode: single image test, and batch test (through a txt file)
    if opt.prompt.endswith('.txt'):
        assert opt.prompt.endswith('.txt')
        image_paths = []
        prompts = []
        with open(opt.prompt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                image_paths.append(line.split('; ')[0])
                prompts.append(line.split('; ')[1])
    else:
        image_paths = [opt.cond_path]
        prompts = [opt.prompt]


    # prepare models
    sd_model, sampler = get_sd_models(opt)
    adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))
    cond_model = None
    if opt.cond_inp_type == 'image':
        cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond))

    process_cond_module = getattr(api, f'get_cond_{which_cond}')

    # inference
    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):
        for test_idx, (cond_path, prompt) in enumerate(zip(image_paths, prompts)):
            seed_everything(opt.seed)
            for v_idx in range(opt.n_samples):
                # colormap
                output_name = os.path.basename(opt.path_img)
                cond = process_cond_module(opt, cond_path, opt.cond_inp_type, cond_model)
                cv2.imwrite(os.path.join(opt.outdir, 'color',output_name.replace('jpg','png')), tensor2img(cond))

                # result
                adapter_features, append_to_context = get_adapter_feature(cond, adapter)
                result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)
                result.save(os.path.join(opt.outdir, output_name))


if __name__ == '__main__':
    main()
