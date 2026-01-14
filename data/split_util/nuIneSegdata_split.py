import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
# ================= 配置区域 =================
# 数据集根目录
data_root = "/mnt/data/zruan/kqy/DATA/NuInsSeg"
pkl_save_path = "/mnt/data/zruan/kqy/icip/lists/nuInsSeg"
# 随机种子 (保证可复现)
SEED = 19
# K-Fold 数量
N_SPLITS = 3
VAL_RATIO = 1/(N_SPLITS*2)
# ===========================================

def calculate_split_counts(n_total, ratio):
    """
    计算验证集和训练集的数量，严格遵守 '要么0，要么>=2' 的规则
    """
    # 1. 初始计算 (向下取整)
    n_val = int(n_total * ratio)
    n_train = n_total - n_val
    
    # 2. 修正逻辑
    # 如果验证集算出是 1，必须处理
    if n_val == 1:
        # 尝试提升为 0
        n_val = 2
        n_train = n_total - n_val
    
    # 3. 安全检查
    # 如果提升后导致训练集小于 2 (比如总共就 3 张，分给 val 2 张，train 剩 1 张)
    # 或者训练集本来就小于 2 (极罕见情况)
    # 策略：弃车保帅，取消验证集，全部给训练集
    if n_train < 2:
        n_val = 0
        n_train = n_total
        
    return n_train, n_val

print(f"🚀 开始执行 {N_SPLITS}-Fold 划分 (严格约束: 数量必须 >= 2 或 = 0)...")

# --- 1. 数据收集 ---
all_image_paths = []
all_labels = []

for class_id in range(1, 32): 
    str_id = str(class_id)
    class_dir = os.path.join(data_root, str_id)
    if not os.path.isdir(class_dir):
        continue
    files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.tif'))]
    for file_name in files:
        all_image_paths.append(os.path.join(class_dir, file_name))
        all_labels.append(class_id)

X = np.array(all_image_paths)
y = np.array(all_labels)

print(f"✅ 数据加载完毕: 总样本 {len(X)}")

# --- 2. Stratified K-Fold ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (train_outer_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'-'*20} Fold {fold} {'-'*20}")
    
    X_train_outer = X[train_outer_idx]
    y_train_outer = y[train_outer_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    final_train_list = []
    final_val_list = []
    
    # --- 3. 内层切分：按类别处理 ---
    unique_classes = np.unique(y_train_outer)
    rng = np.random.RandomState(SEED + fold) # 局部随机
    
    # 用于统计被强制归零或调整的类别
    adjusted_log = []
    
    for cls in unique_classes:
        # 获取该类别在 Outer Train 中的索引
        cls_indices = np.where(y_train_outer == cls)[0]
        rng.shuffle(cls_indices)
        
        n_total_cls = len(cls_indices)
        
        # === 核心调用：计算数量 ===
        n_train_count, n_val_count = calculate_split_counts(n_total_cls, VAL_RATIO)
        
        # 记录调整日志 (方便你看)
        raw_val = int(n_total_cls * VAL_RATIO)
        if n_val_count != raw_val:
            adjusted_log.append(f"Class {cls}: Total {n_total_cls} -> Raw Val {raw_val} -> Adjusted Val {n_val_count}")
        
        # 切分
        val_indices = cls_indices[:n_val_count]
        train_indices = cls_indices[n_val_count:]
        
        # 存入列表
        for idx in train_indices:
            final_train_list.append({"image_path": X_train_outer[idx], "label": int(cls)})
        for idx in val_indices:
            final_val_list.append({"image_path": X_train_outer[idx], "label": int(cls)})

    # 处理 Test Set
    final_test_list = []
    for path, label in zip(X_test, y_test):
        final_test_list.append({"image_path": path, "label": int(label)})

    # --- 4. 保存 ---
    fold_dir = os.path.join(pkl_save_path, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    
    pickle.dump(final_train_list, open(os.path.join(fold_dir, "train.pkl"), "wb"))
    pickle.dump(final_val_list,   open(os.path.join(fold_dir, "val.pkl"), "wb"))
    pickle.dump(final_test_list,  open(os.path.join(fold_dir, "test.pkl"), "wb"))
    
    print(f"💾 保存: Train({len(final_train_list)}) | Val({len(final_val_list)}) | Test({len(final_test_list)})")
    
    if adjusted_log:
        print("   ⚠️ 触发规则调整的类别 (Top 5):")
        for log in adjusted_log[:5]:
            print(f"      {log}")
            
print(f"\n🎉 数据集处理完成！严格遵守 >=2 规则。")