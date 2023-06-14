import json
import cv2
import os
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
            name = file['source']
            name = os.path.basename(name)
            prompt = file['prompt']
            prompt = 'a ' + prompt + ', best quality, extremely detailed'
            self.files.append({'name': name, 'sentence': prompt})

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']

        # print(os.path.join(self.root_path_im, name))
        im = cv2.imread(os.path.join(self.root_path_im, name))
        im = cv2.resize(im, (512, 512))
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        mask = cv2.imread(os.path.join(self.root_path_mask, name.replace('jpg','png')))
        mask = cv2.resize(mask, (512, 512))
        mask = img2tensor(mask, bgr2rgb=True, float32=True) / 255.

        color = cv2.imread(os.path.join(self.root_path_color, name))
        color = cv2.resize(color, (512, 512))
        color = img2tensor(color, bgr2rgb=True, float32=True) / 255.

        sentence = file['sentence']
        return {'im': im, 'mask': mask, 'color': color,'sentence': sentence}

    def __len__(self):
        return len(self.files)
