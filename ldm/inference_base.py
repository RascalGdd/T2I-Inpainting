import argparse
import torch
from omegaconf import OmegaConf
from transformers import AutoFeatureExtractor
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.adapter import Adapter, StyleAdapter, Adapter_light
from ldm.modules.extra_condition.api import ExtraCondition
from ldm.util import fix_cond_shapes, load_model_from_config, read_state_dict
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
DEFAULT_NEGATIVE_PROMPT = 'l'
import PIL.Image as Image
import numpy as np
from einops import repeat


safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    return x_checked_image, has_nsfw_concept

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch

def get_base_argument_parser() -> argparse.ArgumentParser:
    """get the base argument parser for inference scripts"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outdir',
        type=str,
        help='dir to write results to',
        default=None,
    )

    parser.add_argument(
        '--prompt',
        type=str,
        nargs='?',
        default=None,
        help='positive prompt',
    )

    parser.add_argument(
        '--neg_prompt',
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help='negative prompt',
    )

    parser.add_argument(
        '--cond_path',
        type=str,
        default=None,
        help='condition image path',
    )

    parser.add_argument(
        '--cond_inp_type',
        type=str,
        default='image',
        help='the type of the input condition image, take depth T2I as example, the input can be raw image, '
        'which depth will be calculated, or the input can be a directly a depth map image',
    )

    parser.add_argument(
        '--sampler',
        type=str,
        default='ddim',
        choices=['ddim', 'plms'],
        help='sampling algorithm, currently, only ddim and plms are supported, more are on the way',
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='number of sampling steps',
    )

    parser.add_argument(
        '--sd_ckpt',
        type=str,
        default='models/sd-v1-4.ckpt',
        help='path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported',
    )

    parser.add_argument(
        '--vae_ckpt',
        type=str,
        default=None,
        help='vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded',
    )

    parser.add_argument(
        '--adapter_ckpt',
        type=str,
        default=None,
        help='path to checkpoint of adapter',
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/stable-diffusion/sd-v1-inference.yaml',
        help='path to config which constructs SD model',
    )

    parser.add_argument(
        '--max_resolution',
        type=float,
        default=512 * 512,
        help='max image height * width, only for computer with limited vram',
    )

    parser.add_argument(
        '--resize_short_edge',
        type=int,
        default=None,
        help='resize short edge of the input image, if this arg is set, max_resolution will not be used',
    )

    parser.add_argument(
        '--C',
        type=int,
        default=4,
        help='latent channels',
    )

    parser.add_argument(
        '--f',
        type=int,
        default=8,
        help='downsampling factor',
    )

    parser.add_argument(
        '--scale',
        type=float,
        default=7.5,
        help='unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))',
    )

    parser.add_argument(
        '--cond_tau',
        type=float,
        default=1.0,
        help='timestamp parameter that determines until which step the adapter is applied, '
        'similar as Prompt-to-Prompt tau',
    )

    parser.add_argument(
        '--style_cond_tau',
        type=float,
        default=1.0,
        help='timestamp parameter that determines until which step the adapter is applied, '
             'similar as Prompt-to-Prompt tau',
    )

    parser.add_argument(
        '--cond_weight',
        type=float,
        default=1.0,
        help='the adapter features are multiplied by the cond_weight. The larger the cond_weight, the more aligned '
        'the generated image and condition will be, but the generated quality may be reduced',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )

    parser.add_argument(
        '--n_samples',
        type=int,
        default=4,
        help='# of samples to generate',
    )

    return parser


def get_sd_models(opt):
    """
    build stable diffusion model, sampler
    """
    # SD
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, opt.sd_ckpt, opt.vae_ckpt)
    sd_model = model.to(opt.device)

    # sampler
    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(model)
    else:
        raise NotImplementedError

    return sd_model, sampler


def get_t2i_adapter_models(opt):
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, opt.sd_ckpt, opt.vae_ckpt)
    adapter_ckpt_path = getattr(opt, f'{opt.which_cond}_adapter_ckpt', None)
    if adapter_ckpt_path is None:
        adapter_ckpt_path = getattr(opt, 'adapter_ckpt')
    adapter_ckpt = read_state_dict(adapter_ckpt_path)
    new_state_dict = {}
    for k, v in adapter_ckpt.items():
        if not k.startswith('adapter.'):
            new_state_dict[f'adapter.{k}'] = v
        else:
            new_state_dict[k] = v
    m, u = model.load_state_dict(new_state_dict, strict=False)
    if len(u) > 0:
        print(f"unexpected keys in loading adapter ckpt {adapter_ckpt_path}:")
        print(u)

    model = model.to(opt.device)

    # sampler
    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(model)
    else:
        raise NotImplementedError

    return model, sampler


def get_cond_ch(cond_type: ExtraCondition):
    if cond_type == ExtraCondition.sketch or cond_type == ExtraCondition.canny:
        return 1
    return 3


def get_adapters(opt, cond_type: ExtraCondition):
    adapter = {}
    cond_weight = getattr(opt, f'{cond_type.name}_weight', None)
    if cond_weight is None:
        cond_weight = getattr(opt, 'cond_weight')
    adapter['cond_weight'] = cond_weight

    if cond_type == ExtraCondition.style:
        adapter['model'] = StyleAdapter(width=1024, context_dim=768, num_head=8, n_layes=3, num_token=8).to(opt.device)
    elif cond_type == ExtraCondition.color:
        adapter['model'] = Adapter_light(
            cin=64 * get_cond_ch(cond_type),
            channels=[320, 640, 1280, 1280],
            nums_rb=4).to(opt.device)
    else:
        adapter['model'] = Adapter(
            cin=64 * get_cond_ch(cond_type),
            channels=[320, 640, 1280, 1280][:4],
            nums_rb=2,
            ksize=1,
            sk=True,
            use_conv=False).to(opt.device)
    ckpt_path = getattr(opt, f'{cond_type.name}_adapter_ckpt', None)
    if ckpt_path is None:
        ckpt_path = getattr(opt, 'adapter_ckpt')
    state_dict = read_state_dict(ckpt_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('adapter.'):
            new_state_dict[k[len('adapter.'):]] = v
        else:
            new_state_dict[k] = v

    adapter['model'].load_state_dict(new_state_dict)

    return adapter


def diffusion_inference(opt, model, sampler, adapter_features, append_to_context=None):
    path_img = r"/cluster/work/cvl/denfan/diandian/control/T2I-Inpainting/test_images/COD10K-CAM-1-Aquatic-1-BatFish-1.jpg"
    path_mask = r"/cluster/work/cvl/denfan/diandian/control/T2I-Inpainting/test_images/COD10K-CAM-1-Aquatic-1-BatFish-1.png"
    img = Image.open(path_img).convert("RGB").resize([512, 512])
    mask = Image.open(path_mask).resize([512, 512])
    batch = make_batch_sd(img, mask, opt.prompt, "cuda")
    print("prompt is ", opt.prompt)

    # get text embedding
    # c = model.get_learned_conditioning([opt.prompt])
    c = model.cond_stage_model.encode(batch["txt"])

    c_cat = list()
    for ck in model.concat_keys:
        cc = batch[ck].float()
        if ck != model.masked_image_key:
            bchw = [1, 4, 64, 64]
            cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
        else:
            cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
        c_cat.append(cc)
    c_cat = torch.cat(c_cat, dim=1)
    cond = {"c_concat": [c_cat], "c_crossattn": [c]}

    # if opt.scale != 1.0:
    #     uc = model.get_learned_conditioning([opt.neg_prompt])
    # else:
    #     uc = None
    uc_cross = model.get_unconditional_conditioning(1, DEFAULT_NEGATIVE_PROMPT)
    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
    # c, uc = fix_cond_shapes(model, c, uc)

    if not hasattr(opt, 'H'):
        opt.H = 512
        opt.W = 512
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

    samples_latents, _ = sampler.sample(
        S=opt.steps,
        conditioning=cond,
        batch_size=1,
        shape=shape,
        verbose=False,
        unconditional_guidance_scale=opt.scale,
        unconditional_conditioning=uc_full,
        x_T=None,
        features_adapter=adapter_features,
        append_to_context=append_to_context,
        cond_tau=opt.cond_tau,
        style_cond_tau=opt.style_cond_tau,
    )

    x_samples = model.decode_first_stage(samples_latents)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    result = x_samples.cpu().numpy().transpose(0,2,3,1)
    result, has_nsfw_concept = check_safety(result)
    result = result*255

    result = [Image.fromarray(img.astype(np.uint8)) for img in result]

    return result[0]
