from PIL import Image
import os
import glob
from tqdm import tqdm
import cv2
import torchvision.transforms as T
import shutil
import json

input_dir = "F:/ImageSet/kolors_cosplay/fashion_split" 
prefix = "tile image, "

files = glob.glob(f"{input_dir}/**", recursive=True)
image_exts = [".png",".jpg",".jpeg",".webp"]
image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]

print(len(image_files))
for image_file in tqdm(image_files,position=0):
    try:
        text_file = os.path.splitext(image_file)[0] + ".txt"
        # if os.path.exists(text_file):
        #     continue
        
        
        vl2_file = os.path.splitext(image_file)[0] + ".vl2"
        content = ""
        if os.path.exists(vl2_file):
            content = open(vl2_file, encoding="utf-8").read() + " "
            
        wd14_file = os.path.splitext(image_file)[0] + "_wd14.json"
        tags = ""
        character = ""
        if os.path.exists(wd14_file):
            # Open and read the JSON file
            with open(wd14_file, 'r', encoding="utf-8") as file:
                data = json.load(file)
                tags = data["tags"]
                if "character" in data:
                    character = data["character"] + ", "
        
        # caption 
        caption = f"{prefix}{character}{content}{tags} "
        print(caption)
        # break
        # save caption as text file
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(caption)
    except:
        print(image_file)