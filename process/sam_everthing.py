import numpy as np
import os, cv2, torch
from segment_anything import SamAutomaticMaskGenerator, build_sam, SamPredictor, sam_model_registry

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# SAM
sam = sam_model_registry["vit_h"](checkpoint="/opt/checkpoint/sam_vit_h_4b8939.pth")
sam.to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(model=sam, pred_iou_thresh=0.5, stability_score_thresh=0.8)
sam_predictor = SamPredictor(build_sam(checkpoint="/opt/checkpoint/sam_vit_h_4b8939.pth").to(DEVICE))
sam.eval()

# 通过SAM得到每个图片的掩码
def get_mask_by_samauto(image_rgb):
    with torch.no_grad():
        result = mask_generator.generate(image_rgb)
    sorted_result = sorted(result, key=lambda x: x['area'], reverse=True)
    return sorted_result

def get_salient_masks(image_rgb, filename, output_path):
    sam_result = get_mask_by_samauto(image_rgb)
    for index, mask in enumerate(sam_result, start=1):
        mask = np.array(mask["segmentation"], dtype=np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(output_path, f"{filename[:-4]}_{index}.png"), mask * 255)


if __name__ == '__main__':
    images_path = "datasets/excluded/image/"
    output_path = "datasets/excluded/everthing/"
    files = os.listdir(images_path)

    files_len = len(files)
    for index, image_name in enumerate(files, start=1):
        print(f"{index}/{files_len}", image_name)
        file_path = os.path.join(images_path, image_name)
        image_bgr = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        get_salient_masks(image_rgb, image_name, output_path)
