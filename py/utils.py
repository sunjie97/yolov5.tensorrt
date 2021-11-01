import cv2 
import torch 
import torchvision 
import numpy as np 



def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    Input:
        [batch, num_boxes, 85]
    Returns:
        List of detections, on (n, 6) tensor per image [xyxy, conf, cls]
    """
    xc = prediction[..., 4] > conf_thres 

    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000 

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for i, x in enumerate(prediction):

        x = x[xc[i]] 
        
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # cls_conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # number of boxes
        n = x.shape[0]  
        if not n:  # no boxes
            continue
        
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # split the boxes according to corresponding class id 
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        nms_idx = torchvision.ops.nms(boxes, scores, iou_thres) 
        if nms_idx.shape[0] > max_det:  # limit detections
            nms_idx = nms_idx[:max_det]

        output[i] = x[nms_idx]

    return output


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def scale_coords2(img1_shape, coords, img0_shape):
    h_ratio = img1_shape[0] / img0_shape[0]
    w_ratio = img1_shape[1] / img0_shape[1]

    coords[:, [0, 2]] /= w_ratio
    coords[:, [1, 3]] /= h_ratio

    clip_coords(coords, img0_shape)

    return coords 


    

def show(img, predictions, class_names, class_colors, save_path=""):

    for pred in predictions:

        # rectangle 
        p1, p2 = (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3]))
        class_idx = int(pred[-1])
        cv2.rectangle(img, p1, p2, class_colors[class_idx], thickness=2, lineType=cv2.LINE_AA)

        # label 
        label = class_names[class_idx]
        text_w, text_h = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        outside = p1[1] - text_h - 3 >= 0  # label fits outside box 
        p2 = p1[0] + text_w, p1[1] - text_h - 3 if outside else p1[1] + text_h + 3
        cv2.rectangle(img, p1, p2, class_colors[class_idx], -1, cv2.LINE_AA)  # filled 
        cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + text_h + 2), 0, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, img)