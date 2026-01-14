import sys
sys.path.insert(0, "../")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logger import Logger, AverageMeter
from utils.evaluation import Evaluator
from utils import util
from data.dataset import FSSDataset

# --- 模型相关 Import ---
# 请在此处引入您的 Few-Shot 和 Full-Supervision 模型
# from model.your_few_shot_model import FewShotModel
# from model.your_full_sup_model import FullSupModel
from model.few_shot_model import FewShotModel

# =========================================================
# 通用评估函数 (适用于所有任务)
# =========================================================
def evaluate(args, model, dataloader, task_type, fold):    # Force randomness during training / freeze randomness during testing
    util.fix_randseed(args.seed)
    average_meter = AverageMeter(dataloader.dataset)
    Evaluator.initialize()
    
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = util.to_cuda(batch)
            '''
            batch['query_img']: [bsz, C, H, W]
            batch['query_mask']: [bsz, H, W]
            batch['support_imgs']: [bsz, 1, C, H, W]
            batch['support_masks']: [bsz, 1, H, W]
            pred_mask: [bsz, H, W]
            '''
            pred_mask = model(batch)
            assert pred_mask.size() == batch['query_mask'].size()
            # Evaluate prediction
            area_inter, area_union, area_pred, area_gt = Evaluator.classify_prediction(pred_mask.clone(), batch)
            average_meter.update(inter_b=area_inter, union_b=area_union, area_pred_b=area_pred, area_gt_b=area_gt, 
                                    class_id=batch['class_id'], loss=None)
            average_meter.write_process(idx, len(dataloader), epoch=-1, curr_lr=0, write_batch_idx=1)

    # Write evaluation results
    miou, mdice = average_meter.write_result(f'{task_type} Fold-{fold}', 0)
    return miou, mdice


def main():
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Multi-Task Semantic Segmentation based on SAM Variants')
    # 基础参数
    parser.add_argument('--datapath', type=str, default='/mnt/data/zruan/kqy/icip/lists')
    parser.add_argument('--benchmark', type=str, default='pannuke', choices=['nuinsseg', 'pannuke', 'nuinsseg_fixall'])
    parser.add_argument('--imgsize', type=int, default=512, choices=[400, 512, 1024, 408, 1080])
    parser.add_argument('--logpath', type=str, default='test0107')
    parser.add_argument('--bsz', type=int, default=16)  # 1-shot:12 5-shot
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    # 任务控制参数
    parser.add_argument('--task', type=str, default='few_shot', 
                        choices=['zero_shot', 'few_shot', 'full_supervision'],
                        help='Choose the task to run')
    # 模型特定参数
    parser.add_argument('--nshot', type=int, default=1)  
    parser.add_argument('--bpe_path', type=str, default="/mnt/data/zruan/kqy/icip/pretrained/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument('--lr', type=float, default=1e-3)  
    parser.add_argument('--epochs', type=int, default=10, help="Training epochs for full supervision")
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # 定义 3-Fold
    folds = [0, 1, 2] # 假设数据集使用 0,1,2 索引
    fold_results = {'miou': [], 'dice': []}
    
    Logger.info(f"🚀 Starting Task: {args.task} | Benchmark: {args.benchmark}")
    
    # === 3-Fold Cross Validation Loop ===
    for fold in folds:
        print(f"\n{'='*20} Processing Fold {fold} {'='*20}")
        
        # 构建模型 (根据任务初始化不同模型)
        model = None
        if args.task == 'zero_shot':
            print(">>> Initializing SAM3 Model Wrapper...")
            # 这里的 SAM3Wrapper 内部包含了 Sam3Processor, text prompt 生成等逻辑
            pass
        elif args.task == 'few_shot':
            print(">>> Initializing Few-Shot Model...")
            model = FewShotModel(args, "sam3")
            model = model.to("cuda")
            pass
        elif args.task == 'full_supervision':
            print(">>> Initializing Full-Supervision Model...")
            # model = FullSupModel(num_classes=2).cuda()
            pass
        if args.task == 'full_supervision' and not args.load:
            # 构建训练集
            FSSDataset.initialize(img_size=args.imgsize, datapath=args.datapath, split='trn')
            train_loader = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, fold, 'trn')
            # train(args, model, train_loader, fold)
            
        # 3. 评估 (Evaluation)
        print("Evaluating...")
        # 构建测试集
        FSSDataset.initialize(img_size=args.imgsize, datapath=args.datapath, split='test')
        test_loader = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, fold, 'test', shot=args.nshot)
        miou, dice = evaluate(args, model, test_loader, args.task, fold)
        
        print(f"✅ Fold {fold} Result: mIoU = {miou:.4f}, Dice = {dice:.4f}")
        fold_results['miou'].append(miou.detach().cpu().numpy())
        fold_results['dice'].append(dice.detach().cpu().numpy())
        
        # 清理显存
        del model
        torch.cuda.empty_cache()
    
    # === Final Summary ===
    mean_miou = np.mean(fold_results['miou'])
    mean_dice = np.mean(fold_results['dice'])
    
    summary = "\n" + "="*40 + "\n"
    summary += f"FINAL RESULTS ({args.task} @ {args.benchmark})\n"
    summary += "="*40 + "\n"
    for i, fold in enumerate(folds):
        summary += f"Fold {fold}: mIoU = {fold_results['miou'][i]:.2f} | Dice = {fold_results['dice'][i]:.2f}\n"
    summary += "-"*40 + "\n"
    summary += f"MEAN   : mIoU = {mean_miou:.2f} | Dice = {mean_dice:.2f}\n"
    summary += "="*40 + "\n"
    
    print(summary)
    Logger.info(summary)


if __name__ == '__main__':
    main()