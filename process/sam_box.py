import os
import torch
import cv2
import warnings
import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, build_sam, SamPredictor, sam_model_registry

warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# SAM 
sam = sam_model_registry["vit_h"](checkpoint="/opt/checkpoint/sam_vit_h_4b8939.pth")
sam.to(device=device)
sam_predictor = SamPredictor(build_sam(checkpoint="/opt/checkpoint/sam_vit_h_4b8939.pth").to(device))
sam.eval()

# The proportion of boundary pixels was counted
def count_edge_pixels(mask, threshold=0.3):
    rows, cols = mask.shape
    count = 0
    kernel_size = 20
    kernel = np.ones((kernel_size,), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel)

    count += np.sum(mask[0, :])
    count += np.sum(mask[rows - 1, :])
    count += np.sum(mask[1:rows - 1, 0])
    count += np.sum(mask[1:rows - 1, cols - 1])

    count /= 2 * (rows + cols)

    return False if count > threshold else True


def get_mask_by_box(image_path, mask_path):
    off = 50
    image_name = image_path.split("/")[-1]

    image_bgr = cv2.imread(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    sam_predictor.set_image(image)

    mask = cv2.imread(mask_path, 0)
    _, mask = cv2.threshold(mask, 125, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        if area > 1000: areas.append(c)

    bounding_boxes = []
    for i in areas:
        contour = contours[i]
        x, y, w, h = cv2.boundingRect(contour)
        x1 = x + w
        y1 = y + h
        x = x - off if x > off else 0
        y = y - off if y > off else 0
        x1 = x1 + off if 512 - x1 > off else 512
        y1 = y1 + off if 512 - y1 > off else 512

        bounding_boxes.append([x, y, x1, y1])

    input_boxes = torch.tensor(bounding_boxes).to(device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    try:
        with torch.no_grad():
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
    except RuntimeError:
        return np.zeros(image.shape, dtype=np.uint8), 1

    masks = ((masks.sum(dim=0) > 0)[0] * 1).cpu().numpy().astype('uint8')

    if count_edge_pixels(masks):
        return masks * 255, 0
    else:
        return np.zeros(image.shape, dtype=np.uint8), 1


if __name__ == '__main__':
    images_path = "datasets/DUTS/DUTS-TR/DUTS-TR-Image/"
    masks_path = "datasets/DUTS/DUTS-TR/DUTS-TR-Mask/"

    files = os.listdir(images_path)

    black_num = 0
    for index, file_name in enumerate(files):
        png_name = file_name.replace("jpg", "png")
        print("--" * 10, index, png_name, "--" * 10)

        image_path = os.path.join(images_path, file_name)
        mask_path = os.path.join(masks_path, png_name)

        output, num = get_mask_by_box(image_path, mask_path)

        if num == 0:
            cv2.imwrite("datasets/initial/mask" + png_name, output)
        else:
            cv2.imwrite("datasets/excluded/mask/" + png_name, output)
            black_num += num

    print("==" * 20)
    print(black_num)
