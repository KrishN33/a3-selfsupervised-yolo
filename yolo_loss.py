import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        x1 = x/self.S - 0.5 * w
        y1 = y/self.S - 0.5 * h
        x2 = x/self.S + 0.5 * w
        y2 = y/self.S + 0.5 * h
    
        boxes = torch.stack((x1, y1, x2, y2), dim=1)
        return boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : (list) [(tensor) size (-1, 5)]  
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        tar_bbox = self.xywh2xyxy(box_target)
        conv_box1 = self.xywh2xyxy(pred_box_list[0][..., :4])
        conv_box2 = self.xywh2xyxy(pred_box_list[1][..., :4])
    
        b1_iou = compute_iou(conv_box1, tar_bbox)
        b2_iou = compute_iou(conv_box2, tar_bbox)
    
        iou_1 = torch.diagonal(b1_iou)
        iou_2 = torch.diagonal(b2_iou)
    
        mask = iou_1 > iou_2
    
        best_iou = torch.max(iou_1, iou_2)
        best_boxes = torch.where(mask.unsqueeze(-1).expand_as(pred_box_list[0]), pred_box_list[0], pred_box_list[1])
    
        return best_iou, best_boxes
        

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """

        N = classes_pred.size(0)
        
        pred_loss = F.mse_loss(classes_pred[has_object_map], classes_target[has_object_map], reduction = 'sum')

        return pred_loss/N

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """

        loss = 0.0

        no_object = torch.logical_not(has_object_map)
        
        for i in range(self.B):            
            boxes_no_obj = pred_boxes_list[i][no_object][..., 4]
            loss += F.mse_loss(boxes_no_obj, torch.zeros(boxes_no_obj.size()).cuda(), reduction = 'sum')/boxes_no_obj.size(0)
    
        loss *= self.l_noobj
        return loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """

        N = box_pred_conf.size(0)
        loss = F.mse_loss(box_pred_conf, box_target_conf.unsqueeze(1), reduction='sum')

        return loss/N


    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        N = box_pred_response.size(0)
        box_pred_response[..., :2] = torch.sqrt(box_pred_response[..., :2])
        box_target_response[..., :2] = torch.sqrt(box_target_response[..., :2])

        #mse
        
        mse = torch.sum((box_pred_response - box_target_response) ** 2)
    
        reg_loss = self.l_coord * mse
        return reg_loss / N

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0

        pred_boxes_list = [pred_tensor[..., :5], pred_tensor[..., 5:10]]
        pred_cls = pred_tensor[..., 10:]

        cls_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)

        # target_boxes = target_boxes[has_object_map]
        # best_iou, best_boxes = self.find_best_iou_boxes(pred_boxes_list, target_boxes.cuda())
       
        # for i in range(self.B):
        #     pred_boxes_list[i] = pred_boxes_list[i][has_object_map]

        
        target_boxes = target_boxes[has_object_map]
        for i in range(self.B):
            pred_boxes_list[i] = pred_boxes_list[i][has_object_map]
        
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_list, target_boxes.cuda())
        best_ious = best_ious.detach()
        reg_loss = self.get_regression_loss(best_boxes[:, :4],target_boxes)
        
        conf_loss = self.get_contain_conf_loss(best_boxes[:,4:5], best_ious)
        
        total_loss = reg_loss + conf_loss + no_obj_loss + cls_loss

        loss_dict = {
            'total_loss': total_loss,
            'reg_loss': reg_loss,
            'containing_obj_loss': conf_loss,
            'no_obj_loss': no_obj_loss,
            'cls_loss': cls_loss
        }

        return loss_dict