import os
import glob
import pickle
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from collections import defaultdict
from pathlib import Path

class DatasetNuInsSeg(Dataset):
    def __init__(self, datapath, fold, transform, split, shot=1):
        self.split = split
        self.benchmark = 'nuinsseg'
        self.shot = shot
        self.fold = fold
        self.transform = transform
        self.base_path = datapath
        self.class_ids = range(0, 31)

        # 1. 构建 pkl 路径并加载
        pkl_path = os.path.join(datapath, f'NuInsSeg/fold{fold}', f'{split}.pkl')
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.data_list = pickle.load(f)
            
        # 2. 预处理：按类别组织数据，方便采样 Support Set
        self.images_by_class = defaultdict(list)
        for item in self.data_list:
            label = item['label']
            path = item['image_path']
            self.images_by_class[label].append(path)

    def __len__(self):
        return len(self.data_list) 

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs_stack = []
        support_masks_stack = []
        for s_img, s_mask in zip(support_imgs, support_masks):
            s_img = self.transform(s_img)
            support_imgs_stack.append(s_img)
            s_mask = F.interpolate(s_mask.unsqueeze(0).unsqueeze(0).float(), s_img.size()[-2:], mode='nearest').squeeze()
            support_masks_stack.append(s_mask)
        support_imgs = torch.stack(support_imgs_stack)
        support_masks = torch.stack(support_masks_stack)

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

        # --- Mask 路径推断逻辑 ---
        # query_path 可能是 /mnt/.../NuInsSeg/1/image_name.png
        # 我们假设 mask 在 /mnt/.../NuInsSeg/masks/image_name.png
        
        def get_mask_path(img_path):
            p = Path(img_path)
            mask_path = p.parent.parent / 'masks' / p.name
                 
            return mask_path

        query_mask_path = get_mask_path(query_name)
        support_mask_paths = [get_mask_path(p) for p in support_names]

        query_mask = self.read_mask(query_mask_path)
        support_masks = [self.read_mask(p) for p in support_mask_paths]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        # --- 1. 确定 Query ---
        query_item = self.data_list[idx]
        query_path = query_item['image_path']
        class_id = query_item['label']
        
        candidates = self.images_by_class[class_id]

        # 排除 Query 自身 (如果是 1-shot 且总共只有 1 张图，这里会是个问题，但之前的代码保证了 >=2)
        # 即使只有2张图，排除Query后剩1张，正好做1-shot
        potential_supports = [p for p in candidates if p != query_path]
        
        # 如果样本极少（比如刚好只有1个support候选），但我们要 5-shot，就需要允许重复采样
        if len(potential_supports) < self.shot:
            replace = True 
        else:
            replace = False
            
        # 随机采样
        if len(potential_supports) == 0:
            # 极端情况保护：万一某个类真的只有1张图进入了 split
            # 这种情况下只能由 Query 自己充当 Support (虽然会有因为通过，但好过报错)
            support_paths = [query_path] * self.shot
        else:
            support_paths = np.random.choice(potential_supports, self.shot, replace=replace).tolist()

        return query_path, support_paths, class_id - 1
