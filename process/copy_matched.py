import shutil
import os
from tqdm import tqdm

def copy_matching_files(oral_dir, tar_dir, out_dir):
    for file_name in tqdm(os.listdir(tar_dir)):
        dux = '.jpg'
        image_file = os.path.join(oral_dir, os.path.splitext(file_name)[0] + dux)
        out_file = os.path.join(out_dir, os.path.splitext(file_name)[0] + dux)

        if os.path.isfile(image_file):
            shutil.copy(image_file, out_file)
        else:
            print(f"Image file not found for: {file_name}")


if __name__ == '__main__':
    oral_dir = "datasets/DUTS/DUTS-TR/DUTS-TR-Image/"
    tar_dir = "datasets/initial/mask/"
    out_dir = "datasets/initial/image/"
    copy_matching_files(oral_dir, tar_dir, out_dir)
