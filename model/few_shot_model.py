import torch
from torch import nn
from .few_shot.SSP_matching import SSP_MatchingNet
from .sam3.sam3 import build_sam3_image_model
from .sam3.sam3.model.sam3_image_processor import Sam3Processor
from torchvision.transforms import Resize
from PIL import Image
import numpy as np
from .sam3.sam3.model.box_ops import box_xywh_to_cxcywh
from .sam3.sam3.visualization_utils import normalize_bbox
from utils.util import mask_to_boxes_and_save, save_debug_vis


class FewShotModel(nn.Module):
    def __init__(self, args, sam_type="sam3"):
        super(FewShotModel, self).__init__()
        self.imgsize = args.imgsize
        self.fss_model = SSP_MatchingNet("resnet50")
        if sam_type == "sam2":
            pass
        elif sam_type == "sam1":
            pass
        else:
            print("==== sam3 ====")
            self.sam_model = build_sam3_image_model(args.bpe_path)
            self.processor = Sam3Processor(self.sam_model, confidence_threshold=0.5)
    
    def extract_masks_and_logits(self, state):

        if 'masks' not in state or 'masks_logits' not in state:
            return None, None

        masks = state['masks']       # [N, 1, H, W]
        logits = state['masks_logits'] # [N, 1, H, W]
        scores = state.get('scores', [])
        
        if len(scores) == 0:
            return None, None

        # 转 Numpy
        if isinstance(masks, torch.Tensor): masks = masks.cpu().numpy()
        if isinstance(logits, torch.Tensor): logits = logits.cpu().numpy()

        # 维度清洗
        if masks.ndim == 4: masks = masks.squeeze(1)
        if logits.ndim == 4: logits = logits.squeeze(1)

        # 合并所有 Object 的 Mask (OR操作)
        combined_mask = np.any(masks, axis=0) # [H, W]
        
        # 合并 Logits: 取最大值 (Max Pooling) 
        if len(logits) > 0:
            combined_logits = np.max(logits, axis=0) # [H, W]
        else:
            combined_logits = None

        return combined_mask, combined_logits
    
    def predict_sam3(self, query_path, coarse_mask):
        '''
        只能一张张输入，不支持batch
        '''
        img = Image.open(query_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((self.imgsize, self.imgsize), Image.Resampling.LANCZOS)
        inference_state = self.processor.set_image(img)
        self.processor.reset_all_prompts(inference_state)
        # # # prepare text prompt
        # pro_text = "cell"
        # inference_state = self.processor.set_text_prompt(state=inference_state, prompt=pro_text)
        
        # box prompt
        # mask 转 box，使用概率打分过滤
        box_input_xywh = mask_to_boxes_and_save(coarse_mask, save_path="vis/mask_boxes.png")
        if len(box_input_xywh) > 0:
            box_input_cxcywh = box_xywh_to_cxcywh(torch.tensor(box_input_xywh).view(-1, 4))
            norm_boxes_cxcywh = normalize_bbox(box_input_cxcywh, self.imgsize, self.imgsize).tolist()
            box_labels = [True for _ in range(len(box_input_xywh))]
        
            for box, label in zip(norm_boxes_cxcywh, box_labels):
                inference_state = self.processor.add_geometric_prompt(
                    state=inference_state, box=box, label=label
                )
        
        # pred   
        # sam3_mask = inference_state['semantic_seg']
        sam3_mask, sam3_logit = self.extract_masks_and_logits(inference_state)
        if sam3_mask is None:
            sam3_mask = np.zeros((self.imgsize, self.imgsize))
        sam3_mask = torch.from_numpy(sam3_mask).unsqueeze(0).to("cuda").float() 
        
        return img, sam3_mask#, box_input_xywh
        
    def forward(self, batch):
        # obtain dataset
        img_s, img_q, mask_s, mask_q = batch['support_imgs'], batch['query_img'], batch['support_masks'], batch['query_mask']
        
        # Step1: fss
        img_s_list = [img_s[:,i,:,:,:] for i in range(img_s.size()[1])]
        mask_s_list = [mask_s[:,i,:,:] for i in range(mask_s.size()[1])] 
        
        # output [bsz, 2, H, W]
        output = self.fss_model(img_s_list, mask_s_list, img_q, None)[0]
        coarse_mask = torch.argmax(output, dim=1)
        
        # Step2: sam3
        #print("sam3 predict mask")
        results = []
        for i in range(len(batch['query_name'])):
            img_path = batch['query_name'][i]
            c_mask = coarse_mask[i]
            gt_mask = mask_q[i]
            
            # 这里的 sam3_mask 返回的是 [1, H, W] 的 Tensor
            img, sam3_mask_tensor = self.predict_sam3(img_path, c_mask)
            results.append(sam3_mask_tensor)
            
            # --- 可视化逻辑 ---
            # 统一转为 Numpy 格式
            vis_coarse = c_mask.detach().cpu().numpy()
            vis_sam3 = sam3_mask_tensor.squeeze().detach().cpu().numpy()
            vis_gt = gt_mask.detach().cpu().numpy()
            supp_img = Image.open(batch['support_names'][0][i])
            if supp_img.mode != "RGB":
                supp_img = supp_img.convert("RGB")
            vis_support_mask = mask_s_list[0][i].detach().cpu().numpy()
            
            save_debug_vis(
                img, 
                vis_coarse, 
                vis_sam3, 
                vis_gt, 
                supp_img,
                vis_support_mask,
                save_path=f"vis/res_{i}_{batch['class_id'][i].item()}.png"
            )
        
        # [bsz, h, w]
        return torch.cat(results)
        