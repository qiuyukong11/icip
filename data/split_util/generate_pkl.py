import os
import glob
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = '/mnt/data/zruan/kqy/DATA/PanNuke/fold/' 
pkl_save_path = "/mnt/data/zruan/kqy/icip/lists/panNuke"

# 你的文件夹名
FOLDS = ['Fold 1', 'Fold 2', 'Fold 3']

# 指向你的彩色 Mask 文件夹
MASK_DIR_NAME = 'sem_masks' 

OUTPUT_SPLIT_DIR = pkl_save_path

# 你的彩色映射表 (RGB -> ID)
# 我们需要把它反转一下，或者直接用于比较
COLOR_MAP = {
    0: (0, 0, 0),       # Background
    1: (255, 0, 0),     # Neoplastic
    2: (0, 255, 0),     # Inflammatory
    3: (0, 0, 255),     # Connective
    4: (255, 255, 0),   # Dead
    5: (0, 255, 255),   # Epithelial
}
# ===========================================

def rgb_to_label(mask_rgb):
    """
    【核心函数】将 RGB 彩色图 (H, W, 3) 逆向解码为 标签图 (H, W)
    """
    # 转换为 numpy
    mask_np = np.array(mask_rgb) # (H, W, 3)
    
    # 初始化一个全 0 的单通道图 (H, W)
    label_mask = np.zeros(mask_np.shape[:2], dtype=np.uint8)
    
    # 遍历映射表，把颜色换回数字
    for class_id, color in COLOR_MAP.items():
        # 跳过背景 (本来就是0)
        if class_id == 0:
            continue
            
        # 找到所有颜色匹配的像素
        # (mask_np == color) 会返回 (H, W, 3) 的布尔值
        # .all(axis=-1) 确保 R,G,B 三个通道都匹配
        matches = np.all(mask_np == color, axis=-1)
        
        # 赋值
        label_mask[matches] = class_id
        
    return label_mask

def get_image_path_from_mask(mask_path):
    # 1. 替换文件夹
    img_path = mask_path.replace(f'/{MASK_DIR_NAME}/', '/images/')
    
    # 2. 替换前缀 (假设 Mask 是 sem_..., 原图是 img_...)
    dirname, basename = os.path.split(img_path)
    if basename.startswith('sem_'):
        new_basename = basename.replace('sem_', 'img_', 1)
        img_path = os.path.join(dirname, new_basename)
        
    return img_path

def scan_fold_data(fold_name):
    fold_dir = os.path.join(DATA_ROOT, fold_name)
    print(f"🔍 正在扫描 {fold_name} (处理 RGB 彩色 Mask)...")
    
    data_list = []
    search_pattern = os.path.join(fold_dir, MASK_DIR_NAME, '*.png')
    mask_files = glob.glob(search_pattern)
    
    if not mask_files:
        print(f"⚠️  警告: 未找到文件 {search_pattern}")
        return []

    for mask_path in tqdm(mask_files, desc=f"Decoding {fold_name}"):
        try:
            # 1. 打开图片
            img_pil = Image.open(mask_path).convert('RGB') # 强制转 RGB 防止有些图是 RGBA
            
            # 2. 【关键】逆向解码：RGB -> 0,1,2,3,4,5
            mask_label = rgb_to_label(img_pil)
            
            # 3. 统计存在的类别 (排除背景 0)
            unique_classes = np.unique(mask_label)
            valid_classes = [c for c in unique_classes if c != 0]
            
            # 4. 获取 Image 路径
            img_path = get_image_path_from_mask(mask_path)
            
            if not os.path.exists(img_path):
                # 简单容错
                continue
                
            # 5. 生成记录
            for cls in valid_classes:
                data_list.append({
                    'image_path': img_path,
                    'label': int(cls),
                    'fold_origin': fold_name
                })
                
        except Exception as e:
            print(f"Error processing {mask_path}: {e}")
            continue
            
    return data_list

# ================= 主程序 =================

fold_database = {}

# 1. 扫描并解码
for fold in FOLDS:
    fold_database[fold] = scan_fold_data(fold)
    print(f"✅ {fold}: 有效样本数 {len(fold_database[fold])}")

# 2. 实验划分
experiments = {
    'fold0': {'train': 'Fold 1', 'val': 'Fold 2', 'test': 'Fold 3'},
    'fold1': {'train': 'Fold 2', 'val': 'Fold 1', 'test': 'Fold 3'},
    'fold2': {'train': 'Fold 3', 'val': 'Fold 2', 'test': 'Fold 1'}
}

print(f"\n🚀 开始生成 PKL ...")

for exp_name, cfg in experiments.items():
    save_dir = os.path.join(OUTPUT_SPLIT_DIR, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取数据
    train_data = fold_database[cfg['train']]
    val_data   = fold_database[cfg['val']]
    test_data  = fold_database[cfg['test']]
    
    # 保存
    pickle.dump(train_data, open(os.path.join(save_dir, 'train.pkl'), 'wb'))
    pickle.dump(val_data,   open(os.path.join(save_dir, 'val.pkl'), 'wb'))
    pickle.dump(test_data,  open(os.path.join(save_dir, 'test.pkl'), 'wb'))
    
    print(f"💾 [{exp_name}] Saved -> {save_dir}")
    print(f"   Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

print("\n🎉 完成！现在生成的 PKL 包含正确的类别索引了。")