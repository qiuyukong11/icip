对 PanNuke 数据集进行预处理，以实现细胞核实例分割和分类
PanNuke 是一个 H&E 染色图像集，包含来自 19 种不同组织类型的 7,904 个 256 × 256 像素的图像块。细胞核被分为肿瘤细胞、炎症细胞、结缔组织/软组织细胞、坏死细胞和上皮细胞。该数据集分为三个部分：

第一页包含 2,657 张图片
第二页包含 2,524 张图片
折叠3包含2723张图片
更多信息请点击此处

您可以点击此处下载数据集表格。

访问并引用论文TSFD和原始版本

来自原始仓库的示例图像

替代文字

然后基于这三个折叠，将数据集分成三个不同的部分。其中两部分数据用于训练/验证，一部分用于测试集。

预处理
数据下载完成后，您将获得 3 个.zip文件，分别为fold1、fold2和fold3。数据以numpy数组形式存储。提取后的目录结构如下：

📦Fold 1
 ┣ 📂images
 ┃ ┗ 📂fold1
 ┃ ┃ ┣ 📜images.npy
 ┃ ┃ ┗ 📜types.npy
 ┣ 📂masks
 ┃ ┣ 📂fold1
 ┃ ┃ ┗ 📜masks.npy
 ┃ ┣ 📜by-nc-sa.md
 ┃ ┗ 📜README.md
 ┗ 📜README.md
# Fold 2 and 3 also have similar structure
方法一
如果您想使用官方的分割数据，请从process_pannuke_std.py目录运行脚本scripts。只需按如下方式指定输入输出路径即可。

data_dir = '../PanNuke/data/' # location to extracted folds
output_dir = '../Folds/' # location to save op data 
此脚本将创建一个包含三个折叠的目录。数据将转换格式.npy并保存.png。每个折叠的结构如下：

📦Fold 1
 ┣ 📂images
 ┣ 📂inst_masks
 ┗ 📂sem_masks
每个文件夹内的dir文件将分别命名为以下名称；

img_Colon_2_01594.png

inst_Adrenal_gland_2_01041.png

sem_Bile-duct_2_01420.png
第一个词inst表示此掩码包含以边界形式存在的实例信息。
sem意味着它的语义掩码
img指其 H&E 图像
下一个词是Adrenal_gland组织类型
nexr2表示此图像来自原始数据集的第二个折叠。
最后一个数字01041代表图像编号
方法二
此方法中，我按组织类型拆分数据集。因此，数据将根据组织类型保存在 19 个目录中。要生成此数据，请从process_pannuke.py目录中运行脚本scripts。只需按如下方式指定输入/输出路径即可。

data_dir = '../PanNuke/data/' # location to extracted folds
output_dir = '../processed/' # location to save op data 
此脚本将创建一个包含第 19 个子目录的目录，如下所示。

📦processed2
 ┣ 📂Adrenal_gland
 ┣ 📂Bile-duct
 ┣ 📂Bladder
 ┣ 📂Breast
 ┣ 📂Cervix
 ┣ 📂Colon
 ┣ 📂Esophagus
 ┣ 📂HeadNeck
 ┣ 📂Kidney
 ┣ 📂Liver
 ┣ 📂Lung
 ┣ 📂Ovarian
 ┣ 📂Pancreatic
 ┣ 📂Prostate
 ┣ 📂Skin
 ┣ 📂Stomach
 ┣ 📂Testis
 ┣ 📂Thyroid
 ┗ 📂Uterus
数据将被转换格式.npy并保存.png。每个组织目录的结构如下：

 📂Uterus
 ┃ ┣ 📂images
 ┃ ┣ 📂inst_masks
 ┃ ┗ 📂sem_masks
文件命名规则与方法 1 相同。

自定义分割
现在数据已保存到目录中，您可以将数据集拆分为train、val和 等test多个部分。为此，请运行 spliy_pannuke.pyscript form脚本dir.  specify the input/output directories and the split ratio i.e. how much of the data would like to use forval andtest`。

op_dir = '../splits/' # output dir for splits
data_dir = '../processed/' # dir containing the tissue wise splits Method 2


test_split = 0.20 # 20% of total data
val_split = 0.1   # 10% of total data
现在它将op_dir具有以下结构

📦splits
 ┣ 📂test
 ┃ ┣ 📂images
 ┃ ┣ 📂inst_masks
 ┃ ┗ 📂sem_masks
 ┣ 📂train
 ┃ ┣ 📂images
 ┃ ┣ 📂inst_masks
 ┃ ┗ 📂sem_masks
 ┗ 📂val
 ┃ ┣ 📂images
 ┃ ┣ 📂inst_masks
 ┃ ┗ 📂sem_masks
注意：此脚本将按组织类型拆分数据，即每种组织类型的 10% 图像将用于val，20% 用于test。
如果使用上述值运行脚本，则拆分结果如下：

Total Images Found in Adrenal_gland  = 437
========================================
Training Images   = 314
Testing Images    = 88
Validation Images = 35
////////////////////////////////////////
Total Images Found in Bile-duct  = 420
========================================
Training Images   = 302
Testing Images    = 84
Validation Images = 34
////////////////////////////////////////
Total Images Found in Bladder  = 146
========================================
Training Images   = 104
Testing Images    = 30
Validation Images = 12
////////////////////////////////////////
Total Images Found in Breast  = 2351
========================================
Training Images   = 1692
Testing Images    = 471
Validation Images = 188
////////////////////////////////////////
Total Images Found in Cervix  = 293
========================================
Training Images   = 210
Testing Images    = 59
Validation Images = 24
////////////////////////////////////////
Total Images Found in Colon  = 1440
========================================
Training Images   = 1036
Testing Images    = 288
Validation Images = 116
////////////////////////////////////////
Total Images Found in Esophagus  = 424
========================================
Training Images   = 305
Testing Images    = 85
Validation Images = 34
////////////////////////////////////////
Total Images Found in HeadNeck  = 384
========================================
Training Images   = 276
Testing Images    = 77
Validation Images = 31
////////////////////////////////////////
Total Images Found in Kidney  = 134
========================================
Training Images   = 96
Testing Images    = 27
Validation Images = 11
////////////////////////////////////////
Total Images Found in Liver  = 224
========================================
Training Images   = 161
Testing Images    = 45
Validation Images = 18
////////////////////////////////////////
Total Images Found in Lung  = 184
========================================
Training Images   = 132
Testing Images    = 37
Validation Images = 15
////////////////////////////////////////
Total Images Found in Ovarian  = 146
========================================
Training Images   = 104
Testing Images    = 30
Validation Images = 12
////////////////////////////////////////
Total Images Found in Pancreatic  = 195
========================================
Training Images   = 140
Testing Images    = 39
Validation Images = 16
////////////////////////////////////////
Total Images Found in Prostate  = 182
========================================
Training Images   = 130
Testing Images    = 37
Validation Images = 15
////////////////////////////////////////
Total Images Found in Skin  = 187
========================================
Training Images   = 134
Testing Images    = 38
Validation Images = 15
////////////////////////////////////////
Total Images Found in Stomach  = 146
========================================
Training Images   = 104
Testing Images    = 30
Validation Images = 12
////////////////////////////////////////
Total Images Found in Testis  = 196
========================================
Training Images   = 140
Testing Images    = 40
Validation Images = 16
////////////////////////////////////////
Total Images Found in Thyroid  = 226
========================================
Training Images   = 162
Testing Images    = 46
Validation Images = 18
////////////////////////////////////////
Total Images Found in Uterus  = 186
========================================
Training Images   = 133
Testing Images    = 38
Validation Images = 15
////////////////////////////////////////