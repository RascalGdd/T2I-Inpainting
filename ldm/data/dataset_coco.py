import json
import cv2
import os
import torch
import numpy as np
from basicsr.utils import img2tensor
import PIL.Image as Image
import random


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
            
            prompt = file['prompt']
            prompt = 'a ' + prompt + ',realistic and photographic, best quality, extremely detailed'
            self.files.append({'name': name, 'sentence': prompt})

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']

        # print(os.path.join(self.root_path_im, name))
        #im = cv2.imread(os.path.join(self.root_path_im, name))
        #im = cv2.resize(im, (512, 512))
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #im = im.transpose(2,0,1)
        #im = torch.from_numpy(im).to(dtype=torch.float32)/127.5-1.0

        #mask = cv2.imread(os.path.join(self.root_path_mask, name.replace('jpg','png')), cv2.IMREAD_GRAYSCALE)
        #mask = mask.astype(np.float32)/255.0
        #mask = mask[None]
        #mask[mask < 0.5] = 0
        #mask[mask >= 0.5] = 1
        #mask = torch.from_numpy(mask)
    
    
        im = Image.open(os.path.join(self.root_path_im, name)).convert("RGB").resize([512, 512])
        mask = Image.open(os.path.join(self.root_path_mask, name.replace('jpg','png'))).resize([512, 512])    
        im = np.array(im)
        im = im.transpose(2,0,1)
        im = torch.from_numpy(im).to(dtype=torch.float32)/127.5-1.0
    
        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32)/255.0
        mask = mask[None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)    
    
    
        masked_img = im * (mask < 0.5)
        


        
        color = cv2.imread(os.path.join(self.root_path_color, name.replace('jpg','png')))
        color = cv2.resize(color, (512, 512))
        color = img2tensor(color)/ 255.


        sentence = file['sentence']
        return {'im': im, 'mask': mask, 'color': color,'masked_img': masked_img,'sentence': sentence, 'name':name}

    def __len__(self):
        return len(self.files)

class dataset_cod_mask():
    def __init__(self, path_json, root_path_im, root_path_mask, image_size):
        super(dataset_cod_mask, self).__init__()
        data = []
        with open(path_json, 'rt') as f:
            for line in f:
                data = json.loads(line)

        self.files = []
        self.root_path_im = root_path_im
        self.root_path_mask = root_path_mask
        for file in data:
            name = file['source'].replace('\\','/')
            name = os.path.basename(name)
            
            prompt = file['prompt']
            prompt = 'a ' + prompt + ',realistic and photographic, best quality, extremely detailed'
            self.files.append({'name': name, 'sentence': prompt})

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']
        # read img
        im = Image.open(os.path.join(self.root_path_im, name)).convert("RGB").resize([512, 512])  
        im = np.array(im)   
        mask = Image.open(os.path.join(self.root_path_mask, name.replace('jpg','png'))).resize([512, 512])  
        mask = np.array(mask.convert("L"))
        
        # data augmentation
        nonzero_pixels = np.nonzero(mask)
        min_x = np.min(nonzero_pixels[1])
        max_x = np.max(nonzero_pixels[1])
        min_y = np.min(nonzero_pixels[0])
        max_y = np.max(nonzero_pixels[0])  
                     
        left = random.randint(0, min_x)
        right = random.randint(max_x, 512)
        up = random.randint(0, min_y)
        down = random.randint(max_y, 512)
        
        folder = '/cluster/work/cvl/denfan/diandian/control/T2I-COD/test_images/'
        cropped_mask = mask[up:down,left:right]
        cv2.imwrite(os.path.join(folder,'cut', 'mask_'+name), cropped_mask)
        resized_mask = cv2.resize(cropped_mask, (512, 512))
        cv2.imwrite(os.path.join(folder,'cond','mask_'+name), resized_mask)       
        
        cropped_im = im[up:down,left:right,:]
        cv2.imwrite(os.path.join(folder,'cut', 'im_'+name), cv2.cvtColor(cropped_im, cv2.COLOR_RGB2BGR))
        resized_im = cv2.resize(cropped_im, (512, 512))
        cv2.imwrite(os.path.join(folder,'cond','im_'+name), cv2.cvtColor(resized_im, cv2.COLOR_RGB2BGR))  
        
        im = resized_im.transpose(2,0,1)
        im = torch.from_numpy(im).to(dtype=torch.float32)/127.5-1.0        
        
        
        mask = resized_mask.astype(np.float32)/255.0
        mask = mask[None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)    
    
    
        masked_img = im * (mask < 0.5)
        



        sentence = file['sentence']
        return {'im': im, 'mask': mask,'masked_img': masked_img,'sentence': sentence, 'name':name}

    def __len__(self):
        return len(self.files)
