from PIL import Image
import os
import glob
from tqdm import tqdm
import cv2
import torchvision.transforms as T
import shutil
import json

# F:/ImageSet/kolors_cosplay/caption/pixiv/Pixiv Graphics 2022 [NEW]/Zeyu He
input_dir = "F:/ImageSet/kolors_cosplay/train/azami_face"
ori_prefix = "真实照片, realistic phogograph, azami, "

author_names = os.listdir(input_dir)

for author_name in tqdm(author_names,position=0):
    author_dir = f"{input_dir}/{author_name}"
    print('author_dir',author_dir)
    files = glob.glob(f"{author_dir}/**", recursive=True)
    print(len(files))
    image_exts = [".png",".jpg",".jpeg",".webp"]
    image_files = [f for f in files if os.path.splitext(f)[-1].lower() in image_exts]
    prefix = f"{ori_prefix}by {author_name}, "
    print(prefix)
    for image_file in tqdm(image_files,position=1):
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
            print(text_file)
            # break
            # save caption as text file
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(caption)
        except:
            print(image_file)