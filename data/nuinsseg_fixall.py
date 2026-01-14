r""" ISIC few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import json


class DatasetNuInsSegFixed(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=1200):
        self.split = split
        self.benchmark = 'nuinsseg'
        self.shot = shot
        self.num = num
        datapath = "/mnt/data/zruan/kqy/DATA"
        self.base_path = os.path.join(datapath, 'NuInsSeg')
        self.categories = [str(i) for i in range(1, 32)]

        self.class_ids = range(0, 31)
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.categories_class = self.build_categories_class()
        self.img_metadata = self.build_img_metadata()

        self.transform = transform

    def __len__(self):
        return len(self.img_metadata) 

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = query_name.split('/')[-1].split('.')[0]
        ann_path = os.path.join(self.base_path, 'masks')
        query_name = os.path.join(ann_path, query_id) + '.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(ann_path, sid) + '.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        filename = os.path.basename(query_name)          # human_bladder_01.png
        stem = os.path.splitext(filename)[0]       # human_bladder_01
        class_sample = self.categories_class_map["_".join(stem.split("_")[:-1]).replace("_", " ")]
        class_id = int(class_sample) - 1
        
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_id

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            os.path.join(self.base_path, cat)
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata.append(img_path)
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata_classwise[cat] += [img_path]
        return img_metadata_classwise
    
    def build_categories_class(self):
        id_map_path = os.path.join(self.base_path, "folder_id_map.json")
        with open(id_map_path, "r", encoding="utf-8") as f:
            self.categories_class_map = json.load(f)