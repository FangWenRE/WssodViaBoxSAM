import os
import cv2
import itertools
import shutil
from tqdm import tqdm

def calculate_iou(box1, box2):
    # first box
    x1_box1, y1_box1, x2_box1, y2_box1 = box1[0], box1[1], box1[2], box1[3]
    # second box
    x1_box2, y1_box2, x2_box2, y2_box2 = box2[0], box2[1], box2[2], box2[3]

    # calculate the coordinates of the intersection region of the two rectangular boxes
    x_left = max(x1_box1, x1_box2)
    y_top = max(y1_box1, y1_box2)
    x_right = min(x2_box1, x2_box2)
    y_bottom = min(y2_box1, y2_box2)

    # iou
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
    iou = intersection_area / float(area_box1 + area_box2 - intersection_area)

    return iou


def get_boxes(path, limit=None):
    mask = cv2.imread(path, 0)
    _, mask = cv2.threshold(mask, 135, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = {c: cv2.contourArea(contours[c]) for c in range(len(contours)) if cv2.contourArea(contours[c]) > 1000}

    areas = dict(sorted(areas.items(), key=lambda x: x[1], reverse=True))

    if limit is not None:
        areas = dict(itertools.islice(areas.items(), limit))

    bounding_boxes = []
    for i in list(areas.keys()):
        contour = contours[i]
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append([x, y, x + w, y + h])
    return bounding_boxes


def process(gts_path, masks_path, error_path, threshold):
    num = 0
    for file_name in tqdm(os.listdir(masks_path)):
        gt_path = os.path.join(gts_path, file_name)
        mask_path = os.path.join(masks_path, file_name)

        gt_boxes = get_boxes(gt_path)
        gt_boxes_num = len(gt_boxes)

        mask_boxes = get_boxes(mask_path, gt_boxes_num)
        mask_boxes_num = len(mask_boxes)

        all_iou = 0
        iter = min(gt_boxes_num, mask_boxes_num)
        for indx in range(iter):
            iou = calculate_iou(gt_boxes[indx], mask_boxes[indx])
            # print(f"iou:{iou}", gt_boxes[indx],mask_boxes[indx])
            all_iou += iou
        mean_iou = all_iou / (iter + 1e-8)

        # print("mean iou:", mean_iou)
        if iter == 0 or (mean_iou < threshold):
            shutil.move(mask_path, os.path.join(error_path, file_name))
            num += 1
    return num

if __name__ == "__main__":
    gts_path = "datasets/DUTS/DUTS-TR/DUTS-TR-Mask/"
    masks_path = "datasets/initial/mask"
    error_path = "datasets/excluded/mask"

    if not os.path.exists(error_path):
        os.makedirs(error_path)

    threshold = 0.9
    num = process(gts_path, masks_path, error_path, threshold)
    print(num)
