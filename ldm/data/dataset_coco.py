import json
import cv2
import os
import torch
import numpy as np
from basicsr.utils import img2tensor



class dataset_coco_mask_color():
    def __init__(self, path_json, root_path_im, root_path_mask, image_size):
        super(dataset_coco_mask_color, self).__init__()
        with open(path_json, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        data = data['annotations']
        self.files = []
        self.root_path_im = root_path_im
        self.root_path_mask = root_path_mask
        for file in data:
            name = "%012d.png" % file['image_id']
            self.files.append({'name': name, 'sentence': file['caption']})

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']
        # print(os.path.join(self.root_path_im, name))
        im = cv2.imread(os.path.join(self.root_path_im, name.replace('.png', '.jpg')))
        im = cv2.resize(im, (512, 512))
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        mask = cv2.imread(os.path.join(self.root_path_mask, name))  # [:,:,0]
        mask = cv2.resize(mask, (512, 512))
        mask = img2tensor(mask, bgr2rgb=True, float32=True) / 255.  # [0].unsqueeze(0)#/255.

        sentence = file['sentence']
        return {'im': im, 'mask': mask, 'sentence': sentence}

    def __len__(self):
        return len(self.files)


class dataset_cod_mask_color():
    def __init__(self, path_json, root_path_im, root_path_mask,root_path_color, image_size):
        super(dataset_cod_mask_color, self).__init__()
        data = []
        with open(path_json, 'rt') as f:
            for line in f:
                data = json.loads(line)

        self.files = []
        self.root_path_im = root_path_im
        self.root_path_mask = root_path_mask
        self.root_path_color = root_path_color
        for file in data:
            name = file['source'].replace('\\','/')
            name = os.path.basename(name)
            #print('!!!!!!!!!!!',name)
            prompt = file['prompt']
            prompt = 'a ' + prompt + ', best quality, extremely detailed'
            self.files.append({'name': name, 'sentence': prompt})

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']

        # print(os.path.join(self.root_path_im, name))
        im = cv2.imread(os.path.join(self.root_path_im, name))
        im = cv2.resize(im, (512, 512))
        im = im.transpose(2,0,1)
        im = torch.from_numpy(im.astype(np.float32)/ 255.0)
        #im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        mask = cv2.imread(os.path.join(self.root_path_mask, name.replace('jpg','png')), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512))
        #mask = img2tensor(mask, float32=True) / 255.       
        mask = torch.from_numpy(mask.astype(np.float32)/ 255.0)
        
        masked_img = cv2.imread(os.path.join('/cluster/work/cvl/denfan/diandian/control/inpainting/datasets/camo_diff_512/camo_source', name))
        masked_img = cv2.resize(masked_img, (512, 512))
        masked_img = masked_img.transpose(2,0,1)
        #masked_img = img2tensor(masked_img, bgr2rgb=True, float32=True) / 255.
        masked_img = torch.from_numpy(masked_img.astype(np.float32)/ 255.0)
        
        color = cv2.imread(os.path.join(self.root_path_color, name.replace('jpg','png')))
        color = cv2.resize(color, (512, 512))
        color = color.transpose(2,0,1)
        #color = img2tensor(color, bgr2rgb=True, float32=True) / 255.
        color = torch.from_numpy(color.astype(np.float32)/ 255.0)

        sentence = file['sentence']
        return {'im': im, 'mask': mask, 'color': color,'masked_img': masked_img,'sentence': sentence}

    def __len__(self):
        return len(self.files)
