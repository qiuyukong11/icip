r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        if self.benchmark == 'nuinsseg':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 31
        elif self.benchmark == 'pannuke':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 5

        self.intersection_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.union_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.area_pred_buf = torch.zeros([2, self.nclass]).float().cuda() # 新增
        self.area_gt_buf = torch.zeros([2, self.nclass]).float().cuda()   # 新增
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

    def update(self, inter_b, union_b, area_pred_b, area_gt_b, class_id, loss):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())
        # 新增 Dice 相关的累加
        self.area_pred_buf.index_add_(1, class_id, area_pred_b.float())
        self.area_gt_buf.index_add_(1, class_id, area_gt_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou

    def compute_metrics(self):
        """ 计算 mIoU, FB-IoU 和 mDice """
        
        # --- IoU 计算 ---
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        
        # 只取感兴趣的类别
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        # FB-IoU (Foreground-Background IoU, 将所有前景类合并计算)
        # 注意：这里只取了感兴趣的类别进行求和
        inter_sum = self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1)
        union_sum = self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)
        fb_iou = (inter_sum / (union_sum + 1e-10)).mean() * 100

        # --- Dice 计算 ---
        # Dice = 2 * Inter / (Pred + GT)
        dice_denom = self.area_pred_buf + self.area_gt_buf
        dice = 2 * self.intersection_buf.float() / \
               torch.max(torch.stack([dice_denom, self.ones]), dim=0)[0]
        
        dice = dice.index_select(1, self.class_ids_interest)
        mdice = dice[1].mean() * 100

        return miou, fb_iou, mdice

    def write_result(self, split, epoch):
        miou, fb_iou, mdice = self.compute_metrics()

        if len(self.loss_buf) > 0:
            loss_buf = torch.stack(self.loss_buf)
            avg_loss = loss_buf.mean()
        else:
            avg_loss = 0.0

        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % avg_loss
        msg += 'mIoU: %5.2f   ' % miou
        msg += 'FB-IoU: %5.2f   ' % fb_iou
        msg += 'mDice: %5.2f   ' % mdice  # 新增输出
        msg += '***\n'
        
        Logger.info(msg)
        return miou, mdice # 返回给 main 记录

    def write_process(self, batch_idx, datalen, epoch, curr_lr, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            miou, fb_iou, mdice = self.compute_metrics()
            
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            
            if epoch != -1:
                msg += 'Lr: %6.5f  ' % curr_lr
                if len(self.loss_buf) > 0:
                    loss_buf = torch.stack(self.loss_buf)
                    msg += 'L: %6.5f  Avg L: %6.5f  ' % (loss_buf[-1], loss_buf.mean())
            
            msg += 'mIoU: %5.2f  | ' % miou
            msg += 'FB-IoU: %5.2f | ' % fb_iou
            msg += 'mDice: %5.2f' % mdice # 新增输出
            
            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = '_TRAIN_' + logtime if training else '_TEST_' + logtime
        if logpath == '': logpath = logtime

        cls.logpath = os.path.join('logs', logpath)
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath)
        
        logfile = os.path.join(cls.logpath, 'log.txt')

        # ===== 核心修复：手动配置 logger =====
        logger = logging.getLogger()          # root logger
        logger.setLevel(logging.INFO)

        # ❗ 关键：清空所有已有 handler
        logger.handlers.clear()

        # 文件 handler
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setFormatter(logging.Formatter('%(message)s'))

        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== Multi-task Seg. Log ===========')
        logging.info(f'| Task: {getattr(args, "task", "Unknown")}')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']:  # as fc layers are not used in HSNet
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))

