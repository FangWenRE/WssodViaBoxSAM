import numpy as np
import cv2, os, shutil
from tqdm import tqdm

root_path = "datasets/excluded"
mask_path = os.path.join(root_path, "mask")
sm_path =   os.path.join(root_path, "infer")
item_path = os.path.join(root_path, "everthing")
out_path =  os.path.join(root_path, "output")
gt_path = "datasets/DUTS/DUTS-TR/DUTS-TR-Mask/"

if not os.path.exists(out_path):
    os.makedirs(out_path)

def count_edge_pixels(mask, threshold, kernel_size):
    rows, cols = mask.shape
    count = 0
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, 2)

    count += np.sum(mask[0, :])
    count += np.sum(mask[rows - 1, :])
    count += np.sum(mask[1:rows - 1, 0])
    count += np.sum(mask[1:rows - 1, cols - 1])

    count /= 2 * (rows + cols)

    return False if count > threshold else True


NUM = 0
for mask_name in tqdm(os.listdir(mask_path)):
    # print("--"*10, indx, mask_name, "--"*10,end="\r")

    mask = cv2.imread(os.path.join(mask_path, mask_name), 0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    mask = mask / 255

    total = mask if count_edge_pixels(mask, 0.2, 5) else np.zeros(mask.shape, dtype=np.uint8)

    sm = cv2.imread(os.path.join(sm_path, mask_name), 0)
    _, sm = cv2.threshold(sm, 64, 255, cv2.THRESH_BINARY)
    sm = sm / 255

    gt = cv2.imread(os.path.join(gt_path, mask_name), 0)
    _, gt = cv2.threshold(gt, 125, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        areas.append(c)

    bounding_boxes = []
    for i in areas:
        contour = contours[i]
        x, y, w, h = cv2.boundingRect(contour)
        gt = cv2.rectangle(gt, (x, y), (x + w, y + h), (255, 255, 255), thickness=cv2.FILLED)
    gt = gt / 255

    pass_flag = False
    for file_name in os.listdir(item_path):
        file_name_join = "_".join(file_name.split("_")[:-1])
        if file_name_join == mask_name[:-4]:
            item = cv2.imread(os.path.join(item_path, file_name), 0) / 255
            item_sum = np.sum(item)
            area_gt = np.sum(cv2.bitwise_and(item, gt)) / item_sum
            area_sm = np.sum(cv2.bitwise_and(item, sm)) / item_sum
            # print(f"{file_name} > area_mask:{area_gt:.3f} area_sm:{area_sm:.3f}")
            if area_gt > 0.95 and area_sm > 0.3:
                pass_flag = True
                total = total + item

    if pass_flag:
        out = np.multiply(total, gt)
        out = np.uint8(total * 255)
        _, out = cv2.threshold(out, 125, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count_list = []
        for i in contours:
            frame = np.zeros_like(out, dtype=np.uint8)
            cv2.drawContours(frame, [i], -1, (255, 255, 255), thickness=cv2.FILLED)
            frame = np.int64(frame / 255.0)
            gt = np.int64(gt)
            nozero_count = np.sum(cv2.bitwise_and(gt, frame))
            count_list.append(nozero_count)

        sorted_indices = sorted(enumerate(count_list), key=lambda x: x[1], reverse=True)
        top_indices = [index for index, _ in sorted_indices[:len(areas)]]
        out_all = np.zeros_like(out, dtype=np.uint8)
        for item in top_indices:
            cv2.drawContours(out_all, [contours[item]], -1, (255, 255, 255), thickness=cv2.FILLED)

        kernel = np.ones((3, 3), dtype=np.uint8)
        out_all = cv2.dilate(out_all, kernel, 1)
        out_all = cv2.erode(out_all, kernel, 1)
        out_all = np.multiply(out_all, np.uint8(gt))
        cv2.imwrite(os.path.join(out_path, mask_name), out_all)

        NUM += 1

print(NUM)
