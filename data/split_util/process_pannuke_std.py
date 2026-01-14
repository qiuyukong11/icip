import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.segmentation import find_boundaries
import os, glob
from tqdm import trange, tqdm
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

data_dir = '/mnt/data/zruan/kqy/DATA/PanNuke/' # location to extracted folds
output_dir = '/mnt/data/zruan/kqy/DATA/PanNuke/fold/' # location to save o # location to save op data 
# 定义调色盘：Class ID -> (R, G, B)
# PanNuke 通常有 6 类 (0-5)，这里定义了 6 种颜色
# 你可以根据喜好修改颜色 (0-255)
COLOR_MAP = {
    0: (0, 0, 0),       # Class 0 (背景): 黑色
    1: (255, 0, 0),     # Class 1: 红色
    2: (0, 255, 0),     # Class 2: 绿色
    3: (0, 0, 255),     # Class 3: 蓝色
    4: (255, 255, 0),   # Class 4: 黄色
    5: (0, 255, 255),   # Class 5: 青色
}

os.chdir(data_dir)
folds = os.listdir(data_dir)

def get_boundaries(raw_mask):
    '''
    for extracting instance boundaries form the goundtruth file
    '''
    bdr = np.zeros(shape=raw_mask.shape)
    for i in range(raw_mask.shape[-1]-1): # because last chnnel is background
        bdr[:,:,i] = find_boundaries(raw_mask[:,:,i], connectivity=1, mode='thick', background=0)
    bdr = np.sum(bdr, axis = -1)
    return bdr.astype(np.uint8)

def apply_color_map(mask_2d):
    '''将 2D 灰度 Mask (H, W) 映射为 3D 彩色图像 (H, W, 3)'''
    h, w = mask_2d.shape
    # 创建一个全黑的画布
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    unique_ids = np.unique(mask_2d)
    for cls_id in unique_ids:
        if cls_id in COLOR_MAP:
            # 找到等于当前类别的像素，赋予对应颜色
            color_mask[mask_2d == cls_id] = COLOR_MAP[cls_id]
        else:
            # 容错：如果遇到未定义的类别，设为白色
            color_mask[mask_2d == cls_id] = (255, 255, 255)
            
    return color_mask

for i, j in enumerate(folds):
    
    # get paths
    print('Loading Data for {}, Wait...'.format(j))
    img_path =data_dir + j + '/images/fold{}/images.npy'.format(i+1)
    type_path = data_dir + j + '/images/fold{}/types.npy'.format(i+1)
    mask_path = data_dir + j + '/masks/fold{}/masks.npy'.format(i+1)
    print(40*'=', '\n', j, 'Start\n', 40*'=')
    
    # laod numpy files
    masks = np.load(file=mask_path, mmap_mode='r') # read_only mode
    images = np.load(file=img_path, mmap_mode='r') # read_only mode
    types = np.load(file=type_path) 
    
    # creat directories to save images
    try:
        os.mkdir(output_dir + j)
        os.mkdir(output_dir + j + '/images')
        os.mkdir(output_dir + j + '/sem_masks')
        os.mkdir(output_dir + j + '/inst_masks')
    except FileExistsError:
        pass
        
    
    for k in trange(images.shape[0], desc='Writing files for {}'.format(j), total=images.shape[0]):
        
        raw_image =  images[k,:,:,:].astype(np.uint8)
        raw_mask = masks[k,:,:,:]
        sem_mask = np.argmax(raw_mask, axis=-1).astype(np.uint8)
        # swaping channels 0 and 5 so that BG is at 0th channel
        sem_mask = np.where(sem_mask == 5, 6, sem_mask)
        sem_mask = np.where(sem_mask == 0, 5, sem_mask)
        sem_mask = np.where(sem_mask == 6, 0, sem_mask)

        # 【关键步骤】转成彩色
        sem_mask = apply_color_map(sem_mask)

        tissue_type = types[k]
        instances = get_boundaries(raw_mask)
        
        # # for plotting it'll slow down the process considerabelly
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(instances)
        # ax[1].imshow(sem_mask)
        # ax[2].imshow(raw_image)
        
        # save file in op dir
        Image.fromarray(sem_mask).save(output_dir + '/{}/sem_masks/sem_{}_{}_{:05d}.png'.format(j, tissue_type, i+1, k)) 
        Image.fromarray(instances).save(output_dir +'/{}/inst_masks/inst_{}_{}_{:05d}.png'.format(j, tissue_type, i+1, k)) 
        Image.fromarray(raw_image).save(output_dir +'/{}/images/img_{}_{}_{:05d}.png'.format(j, tissue_type, i+1, k)) 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        