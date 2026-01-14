import os
import glob
import pickle
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from collections import defaultdict

# 1. 训练集：第一折；验证集：第二折；测试集：第三折
# 2. 训练集：第 2 折；验证集：第 1 折；测试集：第 3 折。
# 3. 训练集：第3折；验证集：第2折；测试集：第1折
class DatasetPanNuke(Dataset):
    def __init__(self, datapath, fold, transform, split, shot=1):
        self.split = split
        self.benchmark = 'pannuke'
        self.shot = shot
        self.fold = fold
        self.transform = transform
        self.base_path = datapath
        self.class_ids = range(0, 5)
        
        # --- 0. 定义彩色映射 (必须与之前的生成脚本一致) ---
        self.color_map = {
            0: (0, 0, 0),       # Background
            1: (255, 0, 0),     # Neoplastic
            2: (0, 255, 0),     # Inflammatory
            3: (0, 0, 255),     # Connective
            4: (255, 255, 0),   # Dead
            5: (0, 255, 255),   # Epithelial
        }

        # 1. 构建 pkl 路径并加载
        pkl_path = os.path.join(datapath, f'panNuke/fold{fold}', f'{split}.pkl')
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
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names, class_sample)

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

                 'class_id': torch.tensor(class_sample-1)}

        return batch

    def load_frame(self, query_name, support_names, class_id):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        # --- Mask 路径推断逻辑 ---
        def get_mask_path(img_path):
            # 1. 将 /images/ 替换为 /sem_masks/ (因为你要读彩色图)
            mask_path = img_path.replace('/images/', '/sem_masks/')
            
            # 2. 替换文件名前缀 (img_ -> sem_)
            dirname, basename = os.path.split(mask_path)
            if basename.startswith('img_'):
                new_basename = basename.replace('img_', 'sem_', 1)
                mask_path = os.path.join(dirname, new_basename)
            return mask_path

        query_mask_path = get_mask_path(query_name)
        support_mask_paths = [get_mask_path(p) for p in support_names]

        query_mask = self.read_mask(query_mask_path, class_id)
        support_masks = [self.read_mask(p, class_id) for p in support_mask_paths]

        return query_img, query_mask, support_imgs, support_masks
    
    def rgb_to_label(self, mask_rgb_np):
        """
        【关键函数】将 RGB Numpy 图像 (H,W,3) 转换为 Label 图像 (H,W)
        """
        # 初始化一个全0的图
        label_mask = np.zeros(mask_rgb_np.shape[:2], dtype=np.uint8)
        
        # 遍历颜色表进行匹配
        for class_idx, color in self.color_map.items():
            if class_idx == 0: continue # 背景不用管
            
            # 找到像素值匹配 color 的位置
            # (mask_rgb_np == color) 形状是 (H,W,3)
            # .all(axis=-1) 形状是 (H,W)，只有RGB三个通道都对上才是True
            matches = np.all(mask_rgb_np == color, axis=-1)
            label_mask[matches] = class_idx
            
        return label_mask

    def read_mask(self, img_name, target_class):
        """
        读取 RGB Mask -> 转换为 0-5 索引 -> 转换为 Target Class 的二值 Mask
        """
        if not os.path.exists(img_name):
            print(f"Warning: Mask not found: {img_name}")
            return torch.zeros((256, 256)).float()

        # 1. 读取为 RGB
        mask_pil = Image.open(img_name).convert('RGB')
        mask_np_rgb = np.array(mask_pil)
        
        # 2. RGB -> Label (0, 1, 2, 3, 4, 5)
        mask_label = self.rgb_to_label(mask_np_rgb)
        
        # 3. Few-Shot 二值化: 只保留 target_class
        # 结果: Target位置=1.0, 其他位置(背景或其他类)=0.0
        mask_binary = (mask_label == target_class).astype(np.float32)
        
        return torch.from_numpy(mask_binary)

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

        return query_path, support_paths, class_id
